import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

def eval_auc(device, model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_p.append(probs); all_y.append(yb.cpu().numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ap  = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return auc, ap, y, p

def train_epoch(device, model, train_loader, test_loader, opt, loss_fn, epoch, scaler, patience_lim=10, patience=0, best_auc=0.0):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

    auc_tr, ap_tr, _, _ = eval_auc(device, model, train_loader)
    auc_te, ap_te, _, _  = eval_auc(device, model, test_loader)
    print(f"Epoch {epoch:03d} | train AUC {auc_tr:.3f} | test AUC {auc_te:.3f} (AP {ap_te:.3f})")

    # Early stopping
    if auc_te > best_auc + 1e-4:
        best_auc, patience = auc_te, 0
        torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
                   "mlp_checkpoint.pt")
    else:
        patience += 1
        if patience >= patience_lim:
            print("Early stop.")
            return True

    return False