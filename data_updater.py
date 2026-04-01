import os, warnings, time
import pandas as pd
import yfinance as yf
import requests
from functools import wraps

warnings.filterwarnings("ignore")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Paths
NIFTY_15M   = os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv")
VIX_15M     = os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv")
NIFTY_DAILY = os.path.join(DATA_DIR, "nifty_daily.csv")
BANKNIFTY   = os.path.join(DATA_DIR, "bank_nifty_daily.csv")
SP500       = os.path.join(DATA_DIR, "sp500_daily.csv")
VIX_DAILY   = os.path.join(DATA_DIR, "vix_daily.csv")
FII_DII     = os.path.join(DATA_DIR, "fii_dii_daily.csv")
PCR         = os.path.join(DATA_DIR, "pcr_daily.csv")
VIX_TERM    = os.path.join(DATA_DIR, "vix_term_daily.csv")
USDINR      = os.path.join(DATA_DIR, "usdinr_daily.csv")
YIELD_SPREAD = os.path.join(DATA_DIR, "yield_spread_daily.csv")
INDA_DAILY   = os.path.join(DATA_DIR, "inda_daily.csv")
EPI_DAILY    = os.path.join(DATA_DIR, "epi_daily.csv")
CRUDE_DAILY  = os.path.join(DATA_DIR, "crude_daily.csv")
GOLD_DAILY   = os.path.join(DATA_DIR, "gold_daily.csv")
USVIX_DAILY  = os.path.join(DATA_DIR, "us_vix_daily.csv")
RELIANCE_DAILY = os.path.join(DATA_DIR, "reliance_daily.csv")
HDFC_DAILY   = os.path.join(DATA_DIR, "hdfc_daily.csv")
EEM_DAILY    = os.path.join(DATA_DIR, "eem_daily.csv")

# New Macro & Sector Paths
CNXIT_DAILY  = os.path.join(DATA_DIR, "cnxit_daily.csv")
CNXAUTO_DAILY = os.path.join(DATA_DIR, "cnxauto_daily.csv")
CNXFMCG_DAILY = os.path.join(DATA_DIR, "cnxfmcg_daily.csv")
CNXMETAL_DAILY = os.path.join(DATA_DIR, "cnxmetal_daily.csv")
DXY_DAILY    = os.path.join(DATA_DIR, "dxy_daily.csv")
NDX_DAILY    = os.path.join(DATA_DIR, "ndx_daily.csv")
COPPER_DAILY = os.path.join(DATA_DIR, "copper_daily.csv")
FUNDAMENTALS = os.path.join(DATA_DIR, "fundamentals.csv")

# Wave 2: More Sectors, Heavyweights, Asian Peers, Bonds & Safe Havens
CNXPHARMA_DAILY = os.path.join(DATA_DIR, "cnxpharma_daily.csv")
CNXENERGY_DAILY = os.path.join(DATA_DIR, "cnxenergy_daily.csv")
CNXINFRA_DAILY  = os.path.join(DATA_DIR, "cnxinfra_daily.csv")
TCS_DAILY       = os.path.join(DATA_DIR, "tcs_daily.csv")
INFY_DAILY      = os.path.join(DATA_DIR, "infy_daily.csv")
ICICI_DAILY     = os.path.join(DATA_DIR, "icici_daily.csv")
ITC_DAILY       = os.path.join(DATA_DIR, "itc_daily.csv")
HSI_DAILY       = os.path.join(DATA_DIR, "hsi_daily.csv")
NIKKEI_DAILY    = os.path.join(DATA_DIR, "nikkei_daily.csv")
SHANGHAI_DAILY  = os.path.join(DATA_DIR, "shanghai_daily.csv")
USTNX_DAILY     = os.path.join(DATA_DIR, "us10y_daily.csv")
SILVER_DAILY    = os.path.join(DATA_DIR, "silver_daily.csv")
NATGAS_DAILY    = os.path.join(DATA_DIR, "natgas_daily.csv")

# ── HELPER: RETRY DECORATOR ──────────────────────────────────────────────────
def retry_on_failure(max_retries=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if i < max_retries - 1:
                        time.sleep(delay * (backoff ** i))
            print(f" Failed after {max_retries} attempts.")
            return None
        return wrapper
    return decorator

# ── LOADING/SAVING ────────────────────────────────────────────────────────────
def _load(path):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def _save(df, path):
    if df.empty: return
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    df.to_csv(path, index=False)

def _append(existing, fresh):
    if existing.empty: return fresh
    if fresh.empty: return existing
    cutoff = existing['date'].max()
    new = fresh[fresh['date'] > cutoff]
    if new.empty: return existing
    return pd.concat([existing, new], ignore_index=True)

# ── YFINANCE FETCHERS ─────────────────────────────────────────────────────────
def _fetch_yf_daily(ticker):
    raw = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=True)
    if raw.empty: return pd.DataFrame()
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    dc = 'datetime' if 'datetime' in raw.columns else 'date'
    raw = raw.rename(columns={dc: 'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    if raw['date'].dt.tz is not None: raw['date'] = raw['date'].dt.tz_localize(None)
    cols = ['date','open','high','low','close','volume']
    present = [c for c in cols if c in raw.columns]
    return raw[present].copy()

def _fetch_yf_15m(ticker):
    raw = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=True)
    if raw.empty: return pd.DataFrame()
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    dc = 'datetime' if 'datetime' in raw.columns else 'date'
    raw = raw.rename(columns={dc: 'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    if raw['date'].dt.tz is not None:
        raw['date'] = raw['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    return raw[['date','open','high','low','close','volume']].copy()

# ── CORE UPDATERS ─────────────────────────────────────────────────────────────
def update_15m(path, ticker):
    print(f"  {ticker} 15m ... ", end="", flush=True)
    ex = _load(path); fr = _fetch_yf_15m(ticker)
    if fr.empty: print("no data"); return
    cm = _append(ex, fr); _save(cm, path)
    print(f"+{len(cm)-len(ex)} rows → {len(cm)} total")
def update_daily(path, ticker):
    print(f"  {ticker} daily ... ", end="", flush=True)
    ex = _load(path); fr = _fetch_yf_daily(ticker)
    if fr.empty: print("no data"); return
    cm = _append(ex, fr); _save(cm, path)
    print(f"+{len(cm)-len(ex)} rows → {len(cm)} total")

def update_usdinr():
    print(f"  USD/INR ... ", end="", flush=True)
    ex = _load(USDINR); fr = _fetch_yf_daily("INR=X")
    if fr.empty: print("no data"); return
    fr = fr[['date', 'close']].rename(columns={'close': 'usdinr'})
    cm = _append(ex, fr); _save(cm, USDINR)
    print(f"+{len(cm)-len(ex)} rows → {len(cm)} total")

def update_yield_spread():
    print(f"  Yield Spread (10Y-2Y) ... ", end="", flush=True)
    t10 = _fetch_yf_daily("^TNX"); t02 = _fetch_yf_daily("^IRX")
    if t10.empty or t02.empty: print("no data"); return
    t10 = t10[['date', 'close']].rename(columns={'close': 'y10'})
    t02 = t02[['date', 'close']].rename(columns={'close': 'y02'})
    merged = t10.merge(t02, on='date')
    merged['spread'] = merged['y10'] - merged['y02']
    ex = _load(YIELD_SPREAD)
    cm = _append(ex, merged[['date','spread']]); _save(cm, YIELD_SPREAD)
    print(f"+{len(cm)-len(ex)} rows → {len(cm)} total")

@retry_on_failure(max_retries=3)
def _fetch_fii_nse():
    headers = {'User-Agent':'Mozilla/5.0','Accept':'application/json','Referer':'https://www.nseindia.com/'}
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers, timeout=10)
    time.sleep(1)
    r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=headers, timeout=10)
    if r.status_code != 200: return None
    rows = []
    for item in r.json():
        try:
            dt = pd.to_datetime(item.get('date',''), dayfirst=True)
            fn = float(str(item.get('fiiNet',0)).replace(',','') or 0)
            dn = float(str(item.get('diiNet',0)).replace(',','') or 0)
            if fn != 0: rows.append({'date':dt,'fii_net':fn,'dii_net':dn})
        except: continue
    return pd.DataFrame(rows) if rows else None

def update_fii_dii():
    print(f"  FII/DII ... ", end="", flush=True)
    fr = _fetch_fii_nse()
    if fr is None or fr.empty:
        print("failed")
        return
    ex = _load(FII_DII); cm = _append(ex, fr); _save(cm, FII_DII)
    print(f"+{len(cm)-len(ex)} rows → {len(cm)} total")

@retry_on_failure(max_retries=3)
def _fetch_pcr_nse():
    headers = {'User-Agent':'Mozilla/5.0','Accept':'application/json','Referer':'https://www.nseindia.com/'}
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers, timeout=10)
    time.sleep(1)
    r = s.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", headers=headers, timeout=10)
    if r.status_code != 200: return None
    data = r.json(); p_oi = c_oi = 0
    for it in data.get('records',{}).get('data',[]):
        p_oi += it.get('PE',{}).get('openInterest',0)
        c_oi += it.get('CE',{}).get('openInterest',0)
    return round(p_oi / c_oi, 4) if c_oi > 0 else None

def update_pcr():
    print(f"  PCR ... ", end="", flush=True)
    pcr_val = _fetch_pcr_nse()
    if pcr_val is None:
        pcr_val = 1.0
    today = pd.Timestamp('today').normalize()
    ex = _load(PCR); fr = pd.DataFrame([{'date':today,'pcr':pcr_val}]); cm = _append(ex, fr); _save(cm, PCR)
    print(f"PCR={pcr_val:.3f}  +{len(cm)-len(ex)} total rows")

def rebuild_nifty_daily():
    print(f"  Rebuilding nifty_daily ... ", end="", flush=True)
    if not os.path.exists(NIFTY_15M): print("nifty_15m missing"); return
    m = pd.read_csv(NIFTY_15M); m.columns = [c.lower() for c in m.columns]
    m['date'] = pd.to_datetime(m['date']); m['d'] = m['date'].dt.date
    daily = m.groupby('d').agg(open=('open','first'), high=('high','max'),
                                low=('low','min'), close=('close','last')).reset_index().rename(columns={'d':'date'})
    daily['date'] = pd.to_datetime(daily['date'])
    if os.path.exists(VIX_DAILY):
        vd = pd.read_csv(VIX_DAILY); vd.columns = [c.lower() for c in vd.columns]
        vd['date'] = pd.to_datetime(vd['date'])
        vd = vd.rename(columns={'close':'vix'})[['date','vix']]
        daily = daily.merge(vd, on='date', how='left')
    _save(daily, NIFTY_DAILY)
    print(f"{len(daily)} rows")

def update_vix_term():
    print(f"  VIX term structure ... ", end="", flush=True)
    ex = _load(VIX_DAILY)
    if ex.empty: print("vix_daily empty"); return
    ex['vix_near'] = ex['close'].ewm(span=5, adjust=False).mean()
    ex['vix_far']  = ex['close'].ewm(span=21, adjust=False).mean()
    _save(ex[['date','vix_near','vix_far']], VIX_TERM)
    print("updated")

def update_fundamentals():
    print(f"  Fundamentals (P/E) ... ", end="", flush=True)
    try:
        t = yf.Ticker("INDA")
        pe = t.info.get("trailingPE", 22.0)
    except:
        pe = 22.0
    today = pd.Timestamp('today').normalize()
    ex = _load(FUNDAMENTALS)
    
    # Use trailing fallback if live fetch fails
    if pe == 22.0 and not ex.empty:
        pe = float(ex.iloc[-1]['pe_ratio'])
        
    fr = pd.DataFrame([{'date':today, 'pe_ratio':pe}])
    cm = _append(ex, fr)
    _save(cm, FUNDAMENTALS)
    print(f"PE={pe:.2f}  +{len(cm)-len(ex)} total rows")


def run_update():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n── Moses Data update ─────────────────────────────────")
    update_15m(NIFTY_15M, "^NSEI")
    update_15m(VIX_15M,   "^INDIAVIX")
    update_daily(VIX_DAILY,  "^INDIAVIX")
    update_daily(BANKNIFTY,  "^NSEBANK")
    update_daily(SP500,      "^GSPC")
    update_usdinr()
    update_yield_spread()
    rebuild_nifty_daily()
    update_vix_term()
    update_daily(INDA_DAILY, "INDA")
    update_daily(EPI_DAILY, "EPI")
    update_daily(CRUDE_DAILY, "CL=F")
    update_daily(GOLD_DAILY, "GC=F")
    update_daily(USVIX_DAILY, "^VIX")
    update_daily(RELIANCE_DAILY, "RELIANCE.NS")
    update_daily(HDFC_DAILY, "HDFCBANK.NS")
    update_daily(EEM_DAILY, "EEM")
    
    # New Broad Market Sectors & Global Macro
    time.sleep(1) # Prevent yfinance rate limiting
    update_daily(CNXIT_DAILY, "^CNXIT")
    update_daily(CNXAUTO_DAILY, "^CNXAUTO")
    update_daily(CNXFMCG_DAILY, "^CNXFMCG")
    update_daily(CNXMETAL_DAILY, "^CNXMETAL")
    update_daily(DXY_DAILY, "DX-Y.NYB")
    update_daily(NDX_DAILY, "^NDX")
    update_daily(COPPER_DAILY, "HG=F")
    
    # Wave 2: More Sectors, Heavyweights, Asian Peers, Bonds & Safe Havens
    time.sleep(1)
    update_daily(CNXPHARMA_DAILY, "^CNXPHARMA")
    update_daily(CNXENERGY_DAILY, "^CNXENERGY")
    update_daily(CNXINFRA_DAILY, "^CNXINFRA")
    update_daily(TCS_DAILY, "TCS.NS")
    update_daily(INFY_DAILY, "INFY.NS")
    update_daily(ICICI_DAILY, "ICICIBANK.NS")
    update_daily(ITC_DAILY, "ITC.NS")
    time.sleep(1)
    update_daily(HSI_DAILY, "^HSI")
    update_daily(NIKKEI_DAILY, "^N225")
    update_daily(SHANGHAI_DAILY, "000001.SS")
    update_daily(USTNX_DAILY, "^TNX")
    update_daily(SILVER_DAILY, "SI=F")
    update_daily(NATGAS_DAILY, "NG=F")
    
    update_pcr()
    update_fundamentals()
    print("── Done ────────────────────────────────────────────────\n")

if __name__ == "__main__":
    run_update()
