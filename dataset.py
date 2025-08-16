# =========================
# Synthetic options dataset
# =========================
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from scipy.special import erf # Import erf from scipy.special

# ---------- Normal helpers (no SciPy required) ----------
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def ndtr(x: np.ndarray | float) -> np.ndarray | float:
    # Standard normal CDF
    # Use math.erf for scalar and scipy.special.erf for numpy array
    return 0.5 * (1.0 + math.erf(x / SQRT2)) if np.isscalar(x) else 0.5 * (1.0 + erf(np.asarray(x) / SQRT2))

def npdf(x: np.ndarray | float) -> np.ndarray | float:
    # Standard normal PDF
    return (1.0 / SQRT2PI) * np.exp(-0.5 * np.square(x))

# ---------- Black–Scholes (price & Greeks, with dividend yield q) ----------
def _d1_d2(S, K, T, r, q, sigma):
    T = np.maximum(np.asarray(T, float), 1e-8)
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    S = np.asarray(S, float); K = np.asarray(K, float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bsm_price(S, K, T, r, q, sigma, cp="C"):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if cp.upper() == "C":
        return disc_q * S * ndtr(d1) - disc_r * K * ndtr(d2)
    else:
        return disc_r * K * ndtr(-d2) - disc_q * S * ndtr(-d1)

def bsm_greeks(S, K, T, r, q, sigma, cp="C"):
    S = np.asarray(S, float); K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 1e-8)
    r = np.asarray(r, float); q = np.asarray(q, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    nd1 = npdf(d1)
    # Delta
    if cp.upper() == "C":
        delta = disc_q * ndtr(d1)
    else:
        delta = disc_q * (ndtr(d1) - 1.0)
    # Gamma
    gamma = disc_q * nd1 / (S * sigma * np.sqrt(T))
    # Vega (per 1.00 vol)
    vega = disc_q * S * nd1 * np.sqrt(T)
    # Theta (per year)
    term1 = -disc_q * S * nd1 * sigma / (2 * np.sqrt(T))
    if cp.upper() == "C":
        theta = term1 - (r * disc_r * K * ndtr(d2)) + (q * disc_q * S * ndtr(d1))
    else:
        theta = term1 + (r * disc_r * K * ndtr(-d2)) - (q * disc_q * S * ndtr(-d1))
    # Rho (per 1.00 rate)
    rho = (T * disc_r * K * ndtr(d2)) if cp.upper() == "C" else (-T * disc_r * K * ndtr(-d2))
    return delta, gamma, vega, theta, rho

# ---------- OCC symbol builder ----------
def occ_symbol(root: str, expiry: pd.Timestamp, right: str, strike: float) -> str:
    """
    OCC format: ROOT + YYMMDD + C/P + STRIKE(8 digits, strike*1000)
    Example: SPY 2025-08-14 C 400.00 -> SPY250814C00400000
    """
    yy = expiry.year % 100
    mm = expiry.month
    dd = expiry.day
    strike_int = int(round(strike * 1000))
    return f"{root.upper()}{yy:02d}{mm:02d}{dd:02d}{right.upper()}{strike_int:08d}"

# ---------- GBM ----------
def gbm_path(
    S0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt_years: float,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_steps)
    drift = (mu - 0.5 * sigma**2) * dt_years
    diff  = sigma * math.sqrt(dt_years) * Z
    log_levels = np.cumsum(drift + diff)
    S = S0 * np.exp(np.concatenate([[0.0], log_levels]))
    return S  # shape: (n_steps+1,)

# ---------- Config dataclass ----------
@dataclass
class SynthConfig:
    root: str = "SPY"
    start_dt: datetime = datetime.now(timezone.utc) - timedelta(days=365)
    periods: int = 365 * 2    # default ~2 bars/day for 4H; adjust below
    freq_hours: int = 4       # 4H bars
    S0: float = 500.0
    mu: float = 0.06          # annual drift
    sigma: float = 0.20       # annual vol
    r: float = 0.03           # risk-free
    q: float = 0.012          # dividend yield
    expiries_dte: tuple[int, ...] = (7, 14, 30, 60)   # days to expiry from each timestamp
    strike_step: float = 1.0
    n_strikes_each_side: int = 5  # ATM ± N strikes
    iv_base: float = 0.22
    iv_skew: float = -0.30        # smile: <0 means OTM calls higher IV; tune as you like
    iv_term: float = 0.10         # term structure bump with sqrt(T)
    noise_sd: float = 0.01        # microstructure noise on price (1%)
    mispricing_prob: float = 0.05 # occasional edge you can learn
    mispricing_bps: float = 50    # 50 bps mispricing injections
    seed: int = 42

# ---------- Underlying generator ----------
def generate_underlying(cfg: SynthConfig) -> pd.DataFrame:
    bars_per_day = int(24 / cfg.freq_hours)
    n_steps = cfg.periods
    dt_years = cfg.freq_hours / (24 * 252)  # ~252 trading days
    path = gbm_path(cfg.S0, cfg.mu, cfg.sigma, n_steps, dt_years, seed=cfg.seed)
    times = pd.date_range(start=cfg.start_dt, periods=n_steps + 1, freq=f"{cfg.freq_hours}h", tz="UTC") # Changed 'H' to 'h'
    under = pd.DataFrame({"datetime": times, "close": path})
    # (Optional) create simple OHLC from close
    under["open"] = under["close"].shift(1).fillna(under["close"])
    jitter = 0.002
    under["high"] = under[["open","close"]].max(axis=1) * (1 + jitter)
    under["low"]  = under[["open","close"]].min(axis=1) * (1 - jitter)
    under["volume"] = (np.random.poisson(lam=1_000 * bars_per_day, size=len(under))).astype(float)
    return under[["datetime","open","high","low","close","volume"]]

# ---------- Synthetic options chain builder ----------
def generate_synthetic_options(cfg: SynthConfig, under_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []

    # round to nearest strike increment
    def round_to_increment(x: float, inc: float) -> float:
        return round(x / inc) * inc

    for _, row in under_df.iterrows():
        ts = pd.Timestamp(row["datetime"])
        S  = float(row["close"])
        atm = round_to_increment(S, cfg.strike_step)

        for dte in cfg.expiries_dte:
            expiry = ts + pd.Timedelta(days=dte)
            T = max((expiry - ts).total_seconds(), 0.0) / (365.0 * 24 * 3600)
            if T <= 0:
                continue

            for right in ("C", "P"):
                for k in range(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1):
                    K = atm + k * cfg.strike_step
                    if K <= 0:
                        continue

                    # IV surface: base * (1 + term)*(1 + skew)
                    m = (K / S) - 1.0  # simple moneyness
                    iv = cfg.iv_base * (1.0 + cfg.iv_term * math.sqrt(T)) * (1.0 + cfg.iv_skew * m)
                    iv = max(iv, 0.05)

                    fair = bsm_price(S, K, T, cfg.r, cfg.q, iv, right)

                    # microstructure: noise + occasional injected mispricing
                    noisy = fair * (1.0 + rng.normal(0.0, cfg.noise_sd))
                    if rng.random() < cfg.mispricing_prob:
                        sign = 1.0 if rng.random() < 0.5 else -1.0
                        noisy *= (1.0 + sign * cfg.mispricing_bps / 1e4)

                    # simple OHLC around noisy close
                    o = noisy * (1.0 + rng.normal(0, 0.002))
                    c = noisy
                    h = max(o, c) * (1.0 + abs(rng.normal(0, 0.003)))
                    l = min(o, c) * (1.0 - abs(rng.normal(0, 0.003)))

                    # volume higher near ATM, decays with |k|
                    lam = 200.0 * math.exp(-0.4 * abs(k))
                    vol = max(1, int(rng.poisson(lam)))

                    symbol = occ_symbol(cfg.root, expiry.tz_convert("UTC"), right, K)

                    rows.append({
                        "symbol": symbol,
                        "datetime": ts,
                        "right": right,
                        "expiry": expiry.tz_convert("UTC"),
                        "strike": float(K),
                        "iv": float(iv),
                        "spot": float(S),
                        "open": float(o),
                        "high": float(h),
                        "low":  float(l),
                        "close": float(c),
                        "volume": float(vol),
                    })

    opt = pd.DataFrame(rows)
    opt.sort_values(["symbol","datetime"], inplace=True)
    opt.reset_index(drop=True, inplace=True)
    return opt

# ---------- Dataset builder (features + 5-day meta-label) ----------
def build_dataset(
    opt_df: pd.DataFrame,
    r: float,
    q: float,
    hold_days: int = 5,
    contract_multiplier: int = 100,
    fee_per_contract: float = 0.65,
    spread_bps: float = 5.0,
) -> pd.DataFrame:
    df = opt_df.copy()
    df = df.sort_values(["symbol","datetime"]).reset_index(drop=True)

    # Time to expiry in years
    T = (df["expiry"].view("int64") - df["datetime"].view("int64")) / 1e9 / (365*24*3600)
    T = np.maximum(T, 1e-8)
    df["T"] = T

    # BSM fair value (using the same IV that generated prices)
    df["bsm_price"] = np.where(
        df["right"].str.upper().values == "C",
        bsm_price(df["spot"].values, df["strike"].values, df["T"].values, r, q, df["iv"].values, "C"),
        bsm_price(df["spot"].values, df["strike"].values, df["T"].values, r, q, df["iv"].values, "P")
    )

    dlt, gmm, vega, theta, rho = bsm_greeks(df["spot"].values, df["strike"].values, df["T"].values, r, q, df["iv"].values, "C")
    # Compute puts separately for delta/theta/rho sign where needed
    is_call = (df["right"].str.upper() == "C").values
    dlt_c, gmm_c, vega_c, th_c, rho_c = bsm_greeks(df["spot"].values, df["strike"].values, df["T"].values, r, q, df["iv"].values, "C")
    dlt_p, gmm_p, vega_p, th_p, rho_p = bsm_greeks(df["spot"].values, df["strike"].values, df["T"].values, r, q, df["iv"].values, "P")
    df["delta"] = np.where(is_call, dlt_c, dlt_p)
    df["gamma"] = np.where(is_call, gmm_c, gmm_p)   # same actually
    df["vega"]  = np.where(is_call, vega_c, vega_p) # same actually
    df["theta"] = np.where(is_call, th_c, th_p)
    df["rho"]   = np.where(is_call, rho_c, rho_p)

    # Mispricing features
    df["mid"] = df["close"]
    df["mispricing"] = df["mid"] - df["bsm_price"]
    df["moneyness"] = df["spot"] / df["strike"]
    df["dte_days"] = (df["expiry"].view("int64") - df["datetime"].view("int64")) / 1e9 / (24*3600)

    # Rolling z-score per symbol on mispricing
    df["mis_mean"] = df.groupby("symbol")["mispricing"].transform(lambda x: x.rolling(30, min_periods=10).mean())
    df["mis_std"]  = df.groupby("symbol")["mispricing"].transform(lambda x: x.rolling(30, min_periods=10).std())
    df["mis_z"] = (df["mispricing"] - df["mis_mean"]) / (df["mis_std"].replace(0, np.nan))
    df["mis_z"] = df["mis_z"].clip(-10, 10)

    # Meta-label: exit at min(hold_days, expiry)
    hold_sec = hold_days * 24 * 3600
    exit_ts = np.minimum(
        df["datetime"].view("int64") + int(hold_sec * 1e9),
        df["expiry"].view("int64")
    )
    df["exit_time_target"] = pd.to_datetime(exit_ts, utc=True)

    def first_at_or_after(group: pd.DataFrame) -> pd.DataFrame:
        times = group["datetime"].values
        targets = group["exit_time_target"].values
        idx = np.searchsorted(times, targets, side="left")
        idx = np.minimum(idx, len(times)-1)
        group["exit_close"] = group["close"].values[idx]
        group["exit_datetime"] = group["datetime"].values[idx]
        return group

    df = df.groupby("symbol", group_keys=False).apply(first_at_or_after)

    # Costs and PnL
    spread_cost = (spread_bps / 1e4) * df["mid"] * contract_multiplier
    fees = fee_per_contract * 2.0
    # Trade side from theory: cheap -> long, rich -> short
    df["side"] = np.sign(-df["mispricing"]).replace(0, 0)
    df["gross_pnl"] = (df["exit_close"] - df["mid"]) * df["side"] * contract_multiplier
    df["net_pnl"] = df["gross_pnl"] - (fees + spread_cost)
    df["label"] = (df["net_pnl"] >= 0).astype(int)

    features = [
        "mispricing","mis_z","delta","gamma","vega","theta","rho",
        "iv","moneyness","dte_days","volume"
    ]
    meta = ["symbol","datetime","expiry","right","strike","spot","mid","bsm_price","exit_datetime","net_pnl","label"]

    out = df[meta + features].dropna().reset_index(drop=True)
    return out

# ---------- Quick demo ----------
if __name__ == "__main__":
    cfg = SynthConfig(
        start_dt=datetime.now(timezone.utc) - timedelta(days=180),  # ~6 months of 4H bars
        periods=180 * 2,   # ~2 bars/day for 4H
        freq_hours=4,
        S0=500.0,
        mu=0.06,
        sigma=0.20,
        r=0.03,
        q=0.012,
        expiries_dte=(7, 14, 30, 60),
        strike_step=1.0,
        n_strikes_each_side=4,
        iv_base=0.22,
        iv_skew=-0.25,
        iv_term=0.10,
        noise_sd=0.01,
        mispricing_prob=0.05,
        mispricing_bps=40,
        seed=123,
    )

    under_df = generate_underlying(cfg)
    opt_df = generate_synthetic_options(cfg, under_df)
    dataset = build_dataset(opt_df, r=cfg.r, q=cfg.q, hold_days=5)

    print("Underlying sample:\n", under_df.head())
    print("\nOptions sample:\n", opt_df.head())
    print("\nModel-ready dataset sample:\n", dataset.head())
    print("\nRows:", len(dataset), " Positives:", int(dataset['label'].sum()))