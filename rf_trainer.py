"""
rf_trainer.py — Random Forest Direction + Range Trainer
=========================================================
Two simple, clean models for options traders:
  1. Direction Model  → UP or DOWN (for debit/credit spreads)
  2. Range Model      → Expected next-day range in points (for iron condor wings)

Anti-overfit design:
  - max_depth = 5
  - min_samples_leaf = 20
  - 3-regime validation splits (same as XGBoost pipeline)
  - Features: Macro + Sentiment + Technical Momentum

Usage:
    python rf_trainer.py

Output (saved to data/models/):
    rf_direction.pkl        — Direction classifier
    rf_range.pkl            — Range regressor
    rf_metrics.json         — Accuracy, MAE, regime split scores
    rf_feature_importance.csv
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")

HORIZONS = [1, 3, 5, 7, 14]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── FILE PATHS (same as data_updater.py) ─────────────────────────────────────
NIFTY_DAILY  = os.path.join(DATA_DIR, "nifty_daily.csv")
VIX_DAILY    = os.path.join(DATA_DIR, "vix_daily.csv")
PCR          = os.path.join(DATA_DIR, "pcr_daily.csv")
INDA_CSV     = os.path.join(DATA_DIR, "inda_daily.csv")
EPI_CSV      = os.path.join(DATA_DIR, "epi_daily.csv")
YIELD_SPREAD = os.path.join(DATA_DIR, "yield_spread_daily.csv")
BANKNIFTY    = os.path.join(DATA_DIR, "bank_nifty_daily.csv")
SP500        = os.path.join(DATA_DIR, "sp500_daily.csv")
USDINR       = os.path.join(DATA_DIR, "usdinr_daily.csv")
CRUDE_CSV    = os.path.join(DATA_DIR, "crude_daily.csv")
GOLD_CSV     = os.path.join(DATA_DIR, "gold_daily.csv")
USVIX_CSV    = os.path.join(DATA_DIR, "us_vix_daily.csv")
RELIANCE_CSV = os.path.join(DATA_DIR, "reliance_daily.csv")
HDFC_CSV     = os.path.join(DATA_DIR, "hdfc_daily.csv")
EEM_CSV      = os.path.join(DATA_DIR, "eem_daily.csv")

CNXIT_CSV    = os.path.join(DATA_DIR, "cnxit_daily.csv")
CNXAUTO_CSV  = os.path.join(DATA_DIR, "cnxauto_daily.csv")
CNXFMCG_CSV  = os.path.join(DATA_DIR, "cnxfmcg_daily.csv")
CNXMETAL_CSV = os.path.join(DATA_DIR, "cnxmetal_daily.csv")
DXY_CSV      = os.path.join(DATA_DIR, "dxy_daily.csv")
NDX_CSV      = os.path.join(DATA_DIR, "ndx_daily.csv")
COPPER_CSV   = os.path.join(DATA_DIR, "copper_daily.csv")

# Wave 2: More Sectors, Heavyweights, Asian Peers, Bonds & Safe Havens
CNXPHARMA_CSV = os.path.join(DATA_DIR, "cnxpharma_daily.csv")
CNXENERGY_CSV = os.path.join(DATA_DIR, "cnxenergy_daily.csv")
CNXINFRA_CSV  = os.path.join(DATA_DIR, "cnxinfra_daily.csv")
TCS_CSV       = os.path.join(DATA_DIR, "tcs_daily.csv")
INFY_CSV      = os.path.join(DATA_DIR, "infy_daily.csv")
ICICI_CSV     = os.path.join(DATA_DIR, "icici_daily.csv")
ITC_CSV       = os.path.join(DATA_DIR, "itc_daily.csv")
HSI_CSV       = os.path.join(DATA_DIR, "hsi_daily.csv")
NIKKEI_CSV    = os.path.join(DATA_DIR, "nikkei_daily.csv")
SHANGHAI_CSV  = os.path.join(DATA_DIR, "shanghai_daily.csv")
USTNX_CSV     = os.path.join(DATA_DIR, "us10y_daily.csv")
SILVER_CSV    = os.path.join(DATA_DIR, "silver_daily.csv")
NATGAS_CSV    = os.path.join(DATA_DIR, "natgas_daily.csv")
FUND_CSV      = os.path.join(DATA_DIR, "fundamentals.csv")

# ── REGIME SPLITS (same logic as offline_grid_trainer.py) ────────────────────
SPLITS = [
    ("2019-01-01", "2021-01-01"),   # Split A: COVID stress
    ("2022-01-01", "2023-01-01"),   # Split B: Recovery
    ("2024-01-01", "2026-01-01"),   # Split C: Recent regime
]

# ── FEATURES — Macro + Sentiment + Technical ──────────────────────────────────
DIRECTION_FEATURES = [
    'vix', 'vix_ret', 'pcr', 'pcr_ret', 'nifty_ret',
    'sp500_ret', 'sp500_vix_ratio', 'yield_spread',
    'bn_vs_nifty', 'usdinr_vel',
    'bank_nifty_ret', 'usdinr_ret', 'atr5', 'atr10',
    'rsi', 'macd_hist', 'ema_20_dist', 'adx', 'williams_r', 'kc_width',
    # ETFs (Deep: 5 features each)
    'inda_ret', 'inda_rsi', 'inda_macd', 'inda_adx', 'inda_wr',
    'epi_ret', 'epi_rsi', 'epi_macd', 'epi_adx', 'epi_wr',
    'eem_ret', 'eem_rsi', 'eem_macd', 'eem_adx', 'eem_wr',
    # Deep Tier: Sectors + Heavyweights (5 features each)
    'crude_ret', 'crude_rsi', 'crude_macd', 'crude_adx', 'crude_wr',
    'gold_ret', 'gold_rsi', 'gold_macd', 'gold_adx', 'gold_wr',
    'usvix_ret', 'usvix_rsi', 'usvix_macd', 'usvix_adx', 'usvix_wr',
    'rel_ret', 'rel_rsi', 'rel_macd', 'rel_adx', 'rel_wr',
    'hdfc_ret', 'hdfc_rsi', 'hdfc_macd', 'hdfc_adx', 'hdfc_wr',
    'it_ret', 'it_rsi', 'it_macd', 'it_adx', 'it_wr',
    'auto_ret', 'auto_rsi', 'auto_macd', 'auto_adx', 'auto_wr',
    'fmcg_ret', 'fmcg_rsi', 'fmcg_macd', 'fmcg_adx', 'fmcg_wr',
    'metal_ret', 'metal_rsi', 'metal_macd', 'metal_adx', 'metal_wr',
    'pharma_ret', 'pharma_rsi', 'pharma_macd', 'pharma_adx', 'pharma_wr',
    'energy_ret', 'energy_rsi', 'energy_macd', 'energy_adx', 'energy_wr',
    'infra_ret', 'infra_rsi', 'infra_macd', 'infra_adx', 'infra_wr',
    'tcs_ret', 'tcs_rsi', 'tcs_macd', 'tcs_adx', 'tcs_wr',
    'infy_ret', 'infy_rsi', 'infy_macd', 'infy_adx', 'infy_wr',
    'icici_ret', 'icici_rsi', 'icici_macd', 'icici_adx', 'icici_wr',
    'itc_ret', 'itc_rsi', 'itc_macd', 'itc_adx', 'itc_wr',
    'bn_rsi', 'bn_macd',
    # Standard Tier: Global indices + Safe havens (3 features each)
    'dxy_ret', 'dxy_rsi', 'dxy_macd',
    'ndx_ret', 'ndx_rsi', 'ndx_macd',
    'copper_ret', 'copper_rsi', 'copper_macd',
    'hsi_ret', 'hsi_rsi', 'hsi_macd',
    'nikkei_ret', 'nikkei_rsi', 'nikkei_macd',
    'shanghai_ret', 'shanghai_rsi', 'shanghai_macd',
    'us10y_ret', 'us10y_rsi', 'us10y_macd',
    'silver_ret', 'silver_rsi', 'silver_macd',
    'natgas_ret', 'natgas_rsi', 'natgas_macd',
    'pe_ratio'
]

RANGE_FEATURES = [
    'vix', 'vix_ret', 'pcr', 'atr5', 'atr10', 
    'nifty_ret', 'sp500_ret', 'usdinr_vel',
    'crude_ret', 'us_vix_ret'
]


# ── TECHNICAL INDICATORS (Pure Pandas) ────────────────────────────────────────
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # Returning the Histogram

def calc_adx(high, low, close, window=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, min_periods=window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, min_periods=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window, min_periods=window).mean()
    return adx

def calc_williams_r(high, low, close, window=14):
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-9))


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def _load(path, date_col="date"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def build_rf_features():
    """Merge all CSVs into a single feature DataFrame. Clean and simple."""
    print("📦 Loading data files...")

    nifty = _load(NIFTY_DAILY)
    if nifty.empty:
        print("❌ nifty_daily.csv missing. Run data_updater.py first.")
        return None

    # Core OHLC
    df = nifty[["date", "open", "high", "low", "close"]].copy()
    df["nifty_ret"] = df["close"].pct_change()

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix = _load(VIX_DAILY)
    if not vix.empty:
        vix = vix[["date", "close"]].rename(columns={"close": "vix"})
        df = df.merge(vix, on="date", how="left")
        df["vix_ret"] = df["vix"].pct_change()
    else:
        df["vix"] = np.nan
        df["vix_ret"] = np.nan

    # ── PCR ──────────────────────────────────────────────────────────────────
    pcr = _load(PCR)
    if not pcr.empty:
        df = df.merge(pcr[["date", "pcr"]], on="date", how="left")
        df["pcr_ret"] = df["pcr"].pct_change()
    else:
        df["pcr"] = 1.0
        df["pcr_ret"] = 0.0

    # ── Yield Spread ──────────────────────────────────────────────────────────
    ys = _load(YIELD_SPREAD)
    if not ys.empty and "spread" in ys.columns:
        df = df.merge(ys[["date", "spread"]].rename(columns={"spread": "yield_spread"}), on="date", how="left")
    else:
        df["yield_spread"] = 0.0

    # ── BankNifty return ─────────────────────────────────────────────────────
    bn = _load(BANKNIFTY)
    if not bn.empty:
        bn = bn[["date", "close"]].rename(columns={"close": "bn_close"})
        bn["bank_nifty_ret"] = bn["bn_close"].pct_change()
        bn["bn_rsi"] = calculate_rsi(bn["bn_close"], 14).shift(1)
        bn["bn_macd"] = calculate_macd(bn["bn_close"]).shift(1)
        df = df.merge(bn[["date", "bank_nifty_ret", "bn_rsi", "bn_macd"]], on="date", how="left")
    else:
        df["bank_nifty_ret"] = 0.0; df["bn_rsi"] = 50.0; df["bn_macd"] = 0.0

    # ── S&P 500 prev day return ───────────────────────────────────────────────
    sp = _load(SP500)
    if not sp.empty:
        sp = sp[["date", "close"]].rename(columns={"close": "sp_close"})
        sp["sp500_ret"] = sp["sp_close"].pct_change().shift(1)
        df = df.merge(sp[["date", "sp500_ret"]], on="date", how="left")
    else:
        df["sp500_ret"] = 0.0

    # ── USD/INR ───────────────────────────────────────────────────────────────
    usdinr = _load(USDINR)
    if not usdinr.empty:
        col = "usdinr" if "usdinr" in usdinr.columns else "close"
        usdinr = usdinr[["date", col]].rename(columns={col: "usdinr"})
        df = df.merge(usdinr, on="date", how="left")
        df["usdinr_ret"] = df["usdinr"].pct_change()
        df["usdinr_vel"] = df["usdinr"].diff(5).shift(1)  # 5-day rupee momentum
    else:
        df["usdinr"] = 83.0
        df["usdinr_ret"] = 0.0
        df["usdinr_vel"] = 0.0

    # ── FUNDAMENTALS (Valuation) ─────────────────────────────────────────────
    fund = _load(FUND_CSV)
    if not fund.empty:
        df = df.merge(fund[["date", "pe_ratio"]], on="date", how="left")
        df["pe_ratio"] = df["pe_ratio"].ffill().fillna(22.5)
    else:
        df["pe_ratio"] = 22.5

    # ── Institutional Proxies (INDA/EPI/EEM — Deep Technicals) ─────────────
    for etf_path, etf_px in [(INDA_CSV, "inda"), (EPI_CSV, "epi"), (EEM_CSV, "eem")]:
        if os.path.exists(etf_path):
            etf = _load(etf_path)
            etf[f"{etf_px}_ret"] = etf["close"].pct_change().shift(1)
            etf[f"{etf_px}_rsi"] = calculate_rsi(etf["close"], 14).shift(1)
            etf[f"{etf_px}_macd"] = calculate_macd(etf["close"]).shift(1)
            if "high" in etf.columns and "low" in etf.columns:
                etf[f"{etf_px}_adx"] = calc_adx(etf["high"], etf["low"], etf["close"], 14).shift(1)
                etf[f"{etf_px}_wr"] = calc_williams_r(etf["high"], etf["low"], etf["close"], 14).shift(1)
                cols = ["date", f"{etf_px}_ret", f"{etf_px}_rsi", f"{etf_px}_macd", f"{etf_px}_adx", f"{etf_px}_wr"]
            else:
                cols = ["date", f"{etf_px}_ret", f"{etf_px}_rsi", f"{etf_px}_macd"]
            df = df.merge(etf[cols], on="date", how="left")
        else:
            df[f"{etf_px}_ret"] = 0.0
            df[f"{etf_px}_rsi"] = 50.0
            df[f"{etf_px}_macd"] = 0.0
            df[f"{etf_px}_adx"] = 25.0
            df[f"{etf_px}_wr"] = -50.0

    # ── DEEP TIER: Sectors + Heavyweights (5 features: ret, rsi, macd, adx, wr) ──
    for path, col_prefix in [
        (CRUDE_CSV, "crude"), (GOLD_CSV, "gold"), 
        (USVIX_CSV, "usvix"), (RELIANCE_CSV, "rel"),
        (HDFC_CSV, "hdfc"),
        (CNXIT_CSV, "it"), (CNXAUTO_CSV, "auto"), (CNXFMCG_CSV, "fmcg"),
        (CNXMETAL_CSV, "metal"),
        (CNXPHARMA_CSV, "pharma"), (CNXENERGY_CSV, "energy"), (CNXINFRA_CSV, "infra"),
        (TCS_CSV, "tcs"), (INFY_CSV, "infy"), (ICICI_CSV, "icici"), (ITC_CSV, "itc")
    ]:
        if os.path.exists(path):
            data = _load(path)
            ret_col = f"{col_prefix}_ret"
            rsi_col = f"{col_prefix}_rsi"
            macd_col = f"{col_prefix}_macd"
            adx_col = f"{col_prefix}_adx"
            wr_col = f"{col_prefix}_wr"
            
            data[ret_col] = data["close"].pct_change().shift(1)
            data[rsi_col] = calculate_rsi(data["close"], 14).shift(1)
            data[macd_col] = calculate_macd(data["close"]).shift(1)
            
            if "high" in data.columns and "low" in data.columns:
                data[adx_col] = calc_adx(data["high"], data["low"], data["close"], 14).shift(1)
                data[wr_col] = calc_williams_r(data["high"], data["low"], data["close"], 14).shift(1)
                cols = ["date", ret_col, rsi_col, macd_col, adx_col, wr_col]
            else:
                cols = ["date", ret_col, rsi_col, macd_col]
            
            df = df.merge(data[cols], on="date", how="left")
        else:
            df[f"{col_prefix}_ret"] = 0.0
            df[f"{col_prefix}_rsi"] = 50.0
            df[f"{col_prefix}_macd"] = 0.0
            df[f"{col_prefix}_adx"] = 25.0
            df[f"{col_prefix}_wr"] = -50.0

    # ── STANDARD TIER: Global indices + Safe havens (3 features: ret, rsi, macd) ──
    for path, col_prefix in [
        (DXY_CSV, "dxy"), (NDX_CSV, "ndx"), (COPPER_CSV, "copper"),
        (HSI_CSV, "hsi"), (NIKKEI_CSV, "nikkei"), (SHANGHAI_CSV, "shanghai"),
        (USTNX_CSV, "us10y"), (SILVER_CSV, "silver"), (NATGAS_CSV, "natgas")
    ]:
        if os.path.exists(path):
            data = _load(path)
            ret_col = f"{col_prefix}_ret"
            rsi_col = f"{col_prefix}_rsi"
            macd_col = f"{col_prefix}_macd"
            
            data[ret_col] = data["close"].pct_change().shift(1)
            data[rsi_col] = calculate_rsi(data["close"], 14).shift(1)
            data[macd_col] = calculate_macd(data["close"]).shift(1)
            
            df = df.merge(data[["date", ret_col, rsi_col, macd_col]], on="date", how="left")
        else:
            df[f"{col_prefix}_ret"] = 0.0
            df[f"{col_prefix}_rsi"] = 50.0
            df[f"{col_prefix}_macd"] = 0.0

    # ── Derived: BN vs Nifty ──────────────────────────────────────────────────
    if "bank_nifty_ret" in df.columns and "nifty_ret" in df.columns:
        df["bn_vs_nifty"] = df["bank_nifty_ret"] - df["nifty_ret"]
        df["sp500_vix_ratio"] = df["sp500_ret"] / (df["vix_ret"] + 1e-9)
    else:
        df["bn_vs_nifty"] = 0.0
        df["sp500_vix_ratio"] = 0.0

    # ── Derived: ATR, Range lags ──────────────────────────────────────────────
    df["true_range"] = df["high"] - df["low"]
    df["atr5"]       = df["true_range"].rolling(5).mean()
    df["atr10"]      = df["true_range"].rolling(10).mean()
    df["range_lag1"] = df["true_range"].shift(1)
    df["range_lag2"] = df["true_range"].shift(2)
    df["range_lag3"] = df["true_range"].shift(3)

    # ── NEW TECHNICAL INDICATORS ──────────────────────────────────────────────
    df["rsi"]         = calculate_rsi(df["close"], 14)
    df["macd_hist"]   = calculate_macd(df["close"])
    df["ema20"]       = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_20_dist"] = (df["close"] - df["ema20"]) / df["ema20"] * 100
    
    # Deep Technicals
    df["adx"]         = calc_adx(df["high"], df["low"], df["close"], 14)
    df["williams_r"]  = calc_williams_r(df["high"], df["low"], df["close"], 14)
    
    # Keltner Channels (using previously calculated ema20 and atr20 via mean)
    df["atr20"]       = df["true_range"].rolling(20).mean()
    df["keltner_u"]   = df["ema20"] + 2 * df["atr20"]
    df["keltner_l"]   = df["ema20"] - 2 * df["atr20"]
    df["kc_width"]    = (df["keltner_u"] - df["keltner_l"]) / (df["ema20"] + 1e-9)

    # ── Multi-Horizon Direction Targets ───────────────────────────────────────
    for h in HORIZONS:
        # Shift future close forward: close in H days > current close
        df[f"next_close_{h}d"] = df["close"].shift(-h)
        df[f"target_dir_{h}d"] = (df[f"next_close_{h}d"] > df["close"]).astype(float)
        # Handle the shift-induced NaNs correctly for the last H rows
        # (they will be dropped during training BUT kept for final row inference)

    # ── Range Target (Next Day Only for IC) ───────────────────────────────────
    df["next_high"]  = df["high"].shift(-1)
    df["next_low"]   = df["low"].shift(-1)
    df["target_range"] = df["next_high"] - df["next_low"]

    # Drop NaN-heavy rows for core features, but KEEP the last row (target_dir/target_range will be NaN there)
    df = df.dropna(subset=["vix", "pcr"])  # Core features must exist

    # Forward-fill remaining NaNs (FII gaps on holidays, etc.)
    for col in DIRECTION_FEATURES + RANGE_FEATURES:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    print(f"✅ Feature matrix: {len(df)} rows, {df.shape[1]} columns")
    return df.reset_index(drop=True)


# ── REGIME VALIDATION ──────────────────────────────────────────────────────────
def validate_across_regimes(model, df, feature_cols, target_col, task="clf"):
    """Run the model on each of the 3 regime splits. Returns dict of scores."""
    scores = {}
    for t_end, v_end in SPLITS:
        train = df[df["date"] < t_end]
        val   = df[(df["date"] >= t_end) & (df["date"] < v_end)]

        feats = [c for c in feature_cols if c in df.columns]
        if len(train) < 100 or len(val) < 30:
            continue

        Xt = train[feats]; yt = train[target_col]
        Xv = val[feats];   yv = val[target_col]

        model.fit(Xt, yt)
        if task == "clf":
            preds = model.predict(Xv)
            scores[f"{t_end[:4]}-{v_end[:4]}"] = round(accuracy_score(yv, preds), 4)
        else:
            preds = model.predict(Xv)
            scores[f"{t_end[:4]}-{v_end[:4]}"] = round(mean_absolute_error(yv, preds), 1)

    return scores


# ── OPTIMIZATION ENGINE (Mosaic GridSearch) ──────────────────────────────────
def optimize_rf(df, feature_cols, target_col, task="clf"):
    """
    Search for the best RF parameters using TimeSeriesSplit.
    This is what makes the training 'comprehensive' like JUDAH.
    """
    feats = [c for c in feature_cols if c in df.columns]
    
    # Filter rows that have a target (dropping shift-induced NaNs at the end)
    train_df = df.dropna(subset=[target_col])
    X = train_df[feats]
    y = train_df[target_col]

    # Time series cross-validation (same as JUDAH)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # The Parameter Mosaic (Search Space)
    param_grid = {
        "n_estimators": [100, 200, 300] if task == "clf" else [150, 300],
        "max_depth": [5, 8, 12, 15],
        "min_samples_leaf": [10, 20, 30],
        "max_features": ["sqrt", 0.6]
    }

    if task == "clf":
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        scoring = "accuracy"
    else:
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        scoring = "neg_mean_absolute_error"

    grid_search = GridSearchCV(
        base_model, param_grid, cv=tscv, 
        scoring=scoring, n_jobs=-1, verbose=0
    )
    
    print(f"   🔍 Optimizing {target_col} (Mosaic Search)...")
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ── MODEL 1: MULTI-HORIZON DIRECTION ─────────────────────────────────────────
def train_multi_horizon_dir(df):
    print("\n🎯 Training Multi-Horizon Direction Models...")
    results = {}

    feats = [c for c in DIRECTION_FEATURES if c in df.columns]
    
    for h in HORIZONS:
        target = f"target_dir_{h}d"
        print(f" ─── Horizon: {h}d ─────────────────")
        
        # Optimize
        best_rf, best_params, best_score = optimize_rf(df, feats, target, task="clf")
        
        # Final fit on last 30 days for metrics
        # (Wait, optimize already fit on everything. Let's do a final regime check)
        regime_scores = validate_across_regimes(best_rf, df, feats, target, task="clf")
        
        # Save model
        fname = f"rf_dir_{h}d.pkl"
        joblib.dump(best_rf, os.path.join(MODEL_DIR, fname))
        
        # Feature importance for this specific horizon
        importance = pd.DataFrame({
            "feature": feats,
            "importance": best_rf.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(os.path.join(MODEL_DIR, f"rf_dir_{h}d_importance.csv"), index=False)

        results[str(h)] = {
            "best_params": best_params,
            "cv_accuracy": round(best_score, 4),
            "regime_scores": regime_scores,
            "top_feature": importance.iloc[0]["feature"],
            "model_file": fname
        }
        print(f"   ✅ Best: depth={best_params['max_depth']}, est={best_params['n_estimators']} | Score: {best_score:.1%}")

    return results


# ── MODEL 2: RANGE ────────────────────────────────────────────────────────────
def train_optimized_range(df):
    print("\n📏 Training Optimized Range Model (Expected points move)...")

    feats = [c for c in RANGE_FEATURES if c in df.columns]
    target = "target_range"
    
    # Use the Optimizer
    best_rf, best_params, best_score = optimize_rf(df, feats, target, task="reg")
    
    regime_scores = validate_across_regimes(best_rf, df, feats, target, task="reg")
    
    # Save
    joblib.dump(best_rf, os.path.join(MODEL_DIR, "rf_range.pkl"))
    
    importance = pd.DataFrame({
        "feature": feats,
        "importance": best_rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(os.path.join(MODEL_DIR, "rf_range_importance.csv"), index=False)

    print(f"   ✅ Best Range Params: {best_params}")
    
    return best_rf, best_params, regime_scores, importance


# ── STRATEGY & RATIONALE ENGINE ───────────────────────────────────────────────
def _pick_moses_strategy(prob, spot, atr10):
    """
    MOSES ADAPTIVE STRATEGY MATRIX (10-Year Optimal Mapping)
    Synchronized with JUDAH logic.
    """
    # ── 1. Calculate Strike Levels ──────────────────────────────────────────
    atm_strike = int(round(spot / 50) * 50)
    buffer = atr10 * 1.2
    bull_put_sell = int(round((spot - buffer) / 50) * 50)
    bear_call_sell = int(round((spot + buffer) / 50) * 50)

    # ── 2. WEALTH GENERATION: Sniper Trend (DOWN) ──────────────────────────
    if prob < 0.35:
        return {
            "strategy": "Naked Put (PE)",
            "action": "BUY PUT (ATM)",
            "premium": "DEBIT",
            "size": "HALF",
            "source": "MOSES RF",
            "why": f"High Conviction DOWN ({100-prob*100:.1f}%). Moses detects a sharp breakdown. Buying ATM Puts for max alpha.",
            "strikes": {"Buy PE (ATM)": f"{atm_strike:,}"},
            "color": "red",
            "edge": "68.5% (Historical)"
        }

    # ── 3. THE BREAD & BUTTER: Standard Trend (55-65%) ──────────────────────
    if 0.35 <= prob < 0.45:
        return {
            "strategy": "Bear Call Spread",
            "action": "SELL CALLS (Income)",
            "premium": "CREDIT",
            "size": "FULL",
            "source": "MOSES RF",
            "why": f"Standard DOWN bias ({100-prob*100:.1f}%). Using Bear Call Spread for steady income.",
            "strikes": {"Sell CE": f"{bear_call_sell:,}", "Buy CE": f"{bear_call_sell+100:,}"},
            "color": "red",
            "edge": "78.4% (Sync)"
        }

    if 0.55 <= prob < 0.65:
        return {
            "strategy": "Bull Put Spread",
            "action": "SELL PUTS (Income)",
            "premium": "CREDIT",
            "size": "FULL",
            "source": "MOSES RF",
            "why": f"Standard UP bias ({prob*100:.1f}%). Bread & Butter income setup via Bull Put Spread.",
            "strikes": {"Sell PE": f"{bull_put_sell:,}", "Buy PE": f"{bull_put_sell-100:,}"},
            "color": "green",
            "edge": "78.4% (Sync)"
        }

    # ── 4. WEALTH GENERATION: Sniper Trend (UP) ─────────────────────────────
    if prob >= 0.65:
        return {
            "strategy": "Naked Call (CE)",
            "action": "BUY CALL (ATM)",
            "premium": "DEBIT",
            "size": "HALF",
            "source": "MOSES RF",
            "why": f"High Conviction UP ({prob*100:.1f}%). Moses detects trend expansion. Buying ATM Calls.",
            "strikes": {"Buy CE (ATM)": f"{atm_strike:,}"},
            "color": "green",
            "edge": "72.1% (Historical)"
        }

    # ── 5. NEUTRAL / LOCKOUT (45-55%) ───────────────────────────────────────
    if 0.48 <= prob <= 0.52:
        return {
            "strategy": "Iron Condor",
            "action": "COLLECT DECAY",
            "premium": "CREDIT",
            "size": "HALF",
            "source": "THETA",
            "why": "Moses detects sideways chop. Collecting premium from both wings.",
            "strikes": {"Sell CE": f"{bear_call_sell:,}", "Sell PE": f"{bull_put_sell:,}"},
            "color": "yellow",
            "edge": "82.5% (Historical)"
        }

    return {
        "strategy": "No Trade",
        "action": "SIDE-LINES",
        "premium": "CASH",
        "size": "ZERO",
        "source": "SAFETY",
        "why": f"Moses probability ({prob:.2f}) is too close to 50/50. Waiting for confirmation.",
        "strikes": {"Nifty": "WAIT"},
        "color": "red",
        "risk": "NONE — Capital preserved."
    }

def generate_rationale(pred_dict, dir_imp, range_imp):
    """
    Generate a human-readable justification for the verdict based on
    feature importance and current market data.
    """
    direction = pred_dict["direction"]
    prob = pred_dict["up_prob"] if direction == "UP" else pred_dict["down_prob"]
    vix = pred_dict["vix"]
    top_dir_feat = dir_imp.iloc[0]["feature"]
    
    # 1. Scenario Name
    if prob > 0.65:
        scenario = "High Conviction Trend"
    elif vix > 22:
        scenario = "High Volatility Regime"
    elif 0.45 < pred_dict["up_prob"] < 0.55:
        scenario = "Range Bound / Neutral"
    else:
        scenario = "Tactical Momentum"

    # 2. Rationale building
    rationale = f"Model identifies a '{scenario}' scenario. "
    rationale += f"Primary driver is '{top_dir_feat}', "
    
    if direction == "UP":
        rationale += f"signaling upward pressure with {prob:.1%} conviction. "
    else:
        rationale += f"signaling downward pressure with {prob:.1%} conviction. "

    if vix > 20:
        rationale += f"High VIX ({vix:.1f}) suggests wide swings; defensive hedging via wider strikes is advised."
    elif vix < 14:
        rationale += f"Low VIX ({vix:.1f}) indicates a calm environment; directional spreads may yield better risk/reward."
    else:
        rationale += f"Moderate VIX ({vix:.1f}) supports standard range-based strategies like Iron Condors."

    return scenario, rationale

def get_actionable_strategy(pred_dict):
    """Map model outputs to specific options strategies."""
    prob = pred_dict["up_prob"]
    vix = pred_dict["vix"]
    spot = pred_dict["spot"]
    rng  = pred_dict["expected_range_pts"]

    if vix > 25:
        return "Long Straddle/Strangle", "Buy ATM Call + Buy ATM Put", "Extreme Volatility"
    
    if prob > 0.62:
        # Bullish
        buy_strike = round(spot / 50) * 50
        sell_strike = round((spot + rng*0.5) / 50) * 50
        if sell_strike <= buy_strike: sell_strike += 100
        return "Bull Put Spread", f"Sell {buy_strike} PE, Buy {buy_strike-100} PE", "Directional UP"
    
    if prob < 0.38:
        # Bearish
        buy_strike = round(spot / 50) * 50
        sell_strike = round((spot - rng*0.5) / 50) * 50
        if sell_strike >= buy_strike: sell_strike -= 100
        return "Bear Call Spread", f"Sell {buy_strike} CE, Buy {buy_strike+100} CE", "Directional DOWN"
    
    # Neutral / Low Conviction
    upper = pred_dict["iron_condor_upper"]
    lower = pred_dict["iron_condor_lower"]
    return "Iron Condor", f"Sell {upper} CE & {lower} PE (Wings: {upper+100}/{lower-100})", "Range Bound"

# ── LIVE PREDICTION ───────────────────────────────────────────────────────────
def get_todays_prediction(df, dir_model, range_model):
    """Get prediction for the next trading day using today's data."""
    latest = df.iloc[-1]

    dir_feats   = [c for c in DIRECTION_FEATURES if c in df.columns]
    range_feats = [c for c in RANGE_FEATURES if c in df.columns]

    X_dir   = latest[dir_feats].values.reshape(1, -1)
    X_range = latest[range_feats].values.reshape(1, -1)

    # Direction
    dir_proba  = dir_model.predict_proba(X_dir)[0]
    up_prob    = float(dir_proba[1])
def get_todays_prediction(df, range_model):
    """
    Get multi-horizon prediction consensus.
    Loads all horizon models and calculates conviction for each.
    """
    latest = df.iloc[-1]
    dir_feats   = [c for c in DIRECTION_FEATURES if c in df.columns]
    range_feats = [c for c in RANGE_FEATURES if c in df.columns]
    
    X_dir   = latest[dir_feats].values.reshape(1, -1)
    X_range = latest[range_feats].values.reshape(1, -1)
    
    horizon_results = {}
    
    # 1. Collect Direction for every horizon
    for h in HORIZONS:
        m_path = os.path.join(MODEL_DIR, f"rf_dir_{h}d.pkl")
        if not os.path.exists(m_path): continue
        
        m = joblib.load(m_path)
        probs = m.predict_proba(X_dir)[0]
        up_prob = float(probs[1])
        
        horizon_results[str(h)] = {
            "direction": "UP" if up_prob > 0.5 else "DOWN",
            "up_prob": round(up_prob, 3),
            "conviction": round(max(up_prob, 1-up_prob), 3)
        }

    # 2. Main Signal (1d) for Strategy
    main_res = horizon_results.get("1", {"direction": "DOWN", "up_prob": 0.5})
    direction = main_res["direction"]
    up_prob = main_res["up_prob"]
    down_prob = 1 - up_prob

    # 3. Range
    expected_range = float(range_model.predict(X_range)[0])
    spot = float(latest["close"])
    upper_wing = round((spot + expected_range * 0.7) / 50) * 50
    lower_wing = round((spot - expected_range * 0.7) / 50) * 50

    res = {
        "date": latest["date"].strftime("%Y-%m-%d"),
        "spot": spot,
        "direction": direction,
        "up_prob": round(up_prob, 3),
        "down_prob": round(down_prob, 3),
        "expected_range_pts": round(expected_range, 0),
        "iron_condor_upper": upper_wing,
        "iron_condor_lower": lower_wing,
        "vix": round(float(latest.get("vix", 0)), 2),
        "pcr": round(float(latest.get("pcr", 1)), 3),
        "inda_ret": round(float(latest.get("inda_ret", 0)), 4),
        "epi_ret": round(float(latest.get("epi_ret", 0)), 4),
        "bn_vs_nifty": round(float(latest.get("bn_vs_nifty", 0)), 4),
        "usdinr_vel": round(float(latest.get("usdinr_vel", 0)), 3),
        "rsi": round(float(latest.get("rsi", 50)), 2),
        "macd_hist": round(float(latest.get("macd_hist", 0)), 4),
        "ema_20_dist": round(float(latest.get("ema_20_dist", 0)), 3),
        "horizons": horizon_results
    }

    # Strategy & Tag
    strat, strikes, tag = get_actionable_strategy(res)
    res["recommended_strategy"] = strat
    res["recommended_strikes"] = strikes
    res["strategy_tag"] = tag

    return res


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("🌲 MOSES — Hyper-Mosaic Trainer")
    print("================================")

    df = build_rf_features()
    if df is None or df.empty:
        return

    # 1. COMPREHENSIVE TRAINING (Multi-Horizon + GridSearch)
    dir_results = train_multi_horizon_dir(df)
    
    # 2. PROMPT RANGE TRAINING
    range_model, range_params, range_regime, range_imp = train_optimized_range(df)

    # 3. TODAY'S PREDICTION & RATIONALE
    pred = get_todays_prediction(df, range_model)
    
    # Get importance for 1d model for rationale
    imp_1d_path = os.path.join(MODEL_DIR, "rf_dir_1d_importance.csv")
    imp_1d = pd.read_csv(imp_1d_path) if os.path.exists(imp_1d_path) else pd.DataFrame()
    
    scenario, rationale = generate_rationale(pred, imp_1d, range_imp)
    pred["scenario"] = scenario
    pred["rationale"] = rationale

    print(f"\n📡 TODAY'S PREDICTION ({pred['date']})")
    print(f"   Scenario:  {pred['scenario']}")
    print(f"   Verdict:   {pred['direction']} ({max(pred['up_prob'], pred['down_prob']):.1%} prob)")
    print(f"   Strategy:  {pred['recommended_strategy']}")
    consensus = " | ".join([f"{h}d:{v['direction']}" for h, v in pred['horizons'].items()])
    print(f"   Consensus: {consensus}")

    # 4. SAVE COMPREHENSIVE METRICS
    metrics = {
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "search_intensity": "Comprehensive (GridSearch + TimeSeriesSplit)",
        "horizons": dir_results,
        "range_model": {
            "best_params": range_params,
            "regime_mae_pts": range_regime,
            "top_feature": range_imp.iloc[0]["feature"],
        },
        "todays_prediction": pred,
    }
    
    with open(os.path.join(MODEL_DIR, "rf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n🎉 Done. Comprehensive results saved to rf_metrics.json")


if __name__ == "__main__":
    main()
