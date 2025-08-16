import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from get_dataset import TabDS
from model import MLP
from train import eval_auc, train_epoch
import math, os, random, numpy as np, pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "mispricing","mis_z","delta","gamma","vega","theta","rho",
    "iv","moneyness","dte_days","volume","strike","spot","is_put"
]
TARGET = "label"
META = ["symbol","datetime","net_pnl"]  # used for threshold/PnL eval

df = pd.read_csv("data/options_mispricinng_data.csv")
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values("datetime").dropna(subset=["right", TARGET]).reset_index(drop=True)

# encode option type
df["is_put"] = (df["right"].str.upper() == "P").astype(float)

# time-based split (80/20 by time)
cut = df["datetime"].quantile(0.80)
train_df = df[df["datetime"] <= cut].copy()
test_df  = df[df["datetime"]  > cut].copy()

# scale numeric features on train only
scaler = StandardScaler()
train_X = scaler.fit_transform(train_df[FEATURES].values)
test_X  = scaler.transform(test_df[FEATURES].values)
train_y = train_df[TARGET].astype(np.float32).values
test_y  = test_df[TARGET].astype(np.float32).values

# pos_weight for BCE (handle imbalance)
pos_w = (len(train_y) - train_y.sum()) / max(train_y.sum(), 1.0)
pos_w = torch.tensor([pos_w], dtype=torch.float32, device=device)

# create datasets and loaders
train_ds = TabDS(train_X, train_y)
test_ds  = TabDS(test_X,  test_y)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=4096, shuffle=False, drop_last=False)

# initialize model
model = MLP(in_dim=len(FEATURES)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# training loop
best_auc = 0.0
best_ckpt_path = "mlp_checkpoint.pt"
patience = 0
patience_lim = 10

for epoch in range(1, 101):
    early_stop = train_epoch(device, model, train_loader, test_loader, opt, loss_fn, epoch, scaler, patience_lim, patience, best_auc)
    if early_stop:
        break
    patience += 1
    if patience >= patience_lim:
        print("Early stopping triggered.")
        break
    current_auc = eval_auc(device, model, test_loader)[0]
    if current_auc is not None and (current_auc > best_auc):
        best_auc = current_auc
        # Save a safe checkpoint dict (state_dict only) whenever AUC improves
        torch.save({"model": model.state_dict()}, best_ckpt_path)
        print(f"✓ Saved best checkpoint to {best_ckpt_path} (AUC={best_auc:.4f})")
    else:
        print(f"No improvement this epoch (AUC={current_auc:.4f}, best={best_auc:.4f})")
    print(f"Best AUC so far: {best_auc:.4f}")
    print(f"Epoch {epoch} completed.")
    print("-" * 50)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"mlp_epoch_{epoch}.pt")
        print(f"Model saved at epoch {epoch}.")
    if epoch == 100:
        print("Training completed after 100 epochs.")
        break
    print(f"Epoch {epoch} | Best AUC: {best_auc:.4f}")
    print("-" * 50)

# ---- Load best checkpoint if present (PyTorch 2.6-safe) ----
if os.path.exists(best_ckpt_path):
    try:
        # In PyTorch 2.6, weights_only defaults to True which can break older pickles.
        # We set weights_only=False since this is our own trusted checkpoint.
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            # If the file itself is a state_dict
            model.load_state_dict(ckpt)
        else:
            # Unexpected format; try loading directly as state_dict
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {best_ckpt_path}.")
    except Exception as e:
        print(f"Could not load checkpoint {best_ckpt_path}: {e}")
else:
    print(f"No checkpoint found at {best_ckpt_path}; skipping load.")

_, _, y_true, p_test = eval_auc(device, model, test_loader)
test_meta = test_df[META].reset_index(drop=True).copy()

# ---- Probability diagnostics + threshold tuning (F1) ----
# Plot histogram to see if probabilities cluster below 0.5
import matplotlib.pyplot as plt
plt.hist(p_test, bins=50)
plt.xlabel("Predicted probability for class 1")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution")
plt.show()

# Use precision-recall curve to find threshold that maximizes F1
prec, rec, th = precision_recall_curve(y_true, p_test)
# thresholds `th` aligns with prec[:-1], rec[:-1]
if len(th) > 0:
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(f1.argmax())
    best_tau = float(th[best_idx])
    print(f"Best threshold by F1: tau={best_tau:.4f} | precision={prec[best_idx]:.3f} | recall={rec[best_idx]:.3f} | F1={f1[best_idx]:.3f}")
else:
    best_tau = 0.50
    print("precision_recall_curve returned no thresholds; defaulting best_tau=0.50")

# We'll use this as a classification decision threshold alongside 0.50 for comparison
tau_decision = best_tau

def pnl_at_threshold(th):
    pick = p_test >= th
    if pick.sum() == 0:
        # No trades selected at this threshold; use -inf so it never beats any real PnL
        return float("-inf"), 0, 0.0
    pnl = test_meta.loc[pick, "net_pnl"].sum()
    return float(pnl), int(pick.sum()), float(pick.mean())

best = {"tau": None, "pnl": float("-inf"), "n": 0, "rate": 0.0}
low = max(0.01, tau_decision - 0.20)
high = min(0.99, tau_decision + 0.20)
for tau in np.linspace(low, high, 41):  # center search around tuned threshold
    pnl, n, rate = pnl_at_threshold(tau)
    if pnl >= best["pnl"]:
        best = {"tau": float(tau), "pnl": float(pnl), "n": int(n), "rate": float(rate)}

if best["tau"] is None:
    # Fallback report at default threshold 0.50
    fallback_tau = 0.50
    pnl, n, rate = pnl_at_threshold(fallback_tau)
    print(f"\nNo profitable threshold found or no trades selected across the grid; "
          f"fallback @tau={fallback_tau:.2f} | trades={n} ({rate*100:.1f}%) | net_pnl={pnl:.2f}")
else:
    print(f"\nBest threshold by net PnL: tau={best['tau']:.2f} "
          f"| trades={best['n']} ({best['rate']*100:.1f}%) | net_pnl={best['pnl']:.2f}")
    print(f"Using tuned decision threshold tau_decision={tau_decision:.4f} for classification metrics.\n")

# ---- Classification reports at fixed 0.50 and tuned tau_decision ----
preds05 = (p_test >= 0.50).astype(int)
print("\nClassification @0.50\n", classification_report(y_true, preds05, digits=3))

preds_best = (p_test >= tau_decision).astype(int)
print(f"\nClassification @best_tau={tau_decision:.2f}\n", classification_report(y_true, preds_best, digits=3))

# ---- Positive-class focused metrics and confusion matrices ----
for name, preds in [("0.50", preds05), (f"{tau_decision:.2f}", preds_best)]:
    prec1, rec1, f1_1, sup1 = precision_recall_fscore_support(y_true, preds, average=None, labels=[1])
    cm = confusion_matrix(y_true, preds, labels=[0,1])
    print(f"\n[Threshold {name}] Positive-class metrics:")
    print(f"  precision_1={prec1[0]:.3f} | recall_1={rec1[0]:.3f} | f1_1={f1_1[0]:.3f} | support_1={int(sup1[0])}")
    print("  Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)