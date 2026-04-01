import os
import pandas as pd

DATA_DIR = r"c:\Users\hp\Desktop\New_ML\Moses-RandomForest\data"
files = [
    "nifty_daily.csv",
    "vix_daily.csv",
    "pcr_daily.csv",
    "fii_dii_daily.csv",
    "yield_spread_daily.csv",
    "bank_nifty_daily.csv",
    "sp500_daily.csv",
    "usdinr_daily.csv"
]

for f in files:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"✅ {f}: {len(df)} rows, columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"❌ {f}: {e}")
    else:
        print(f"❓ {f} missing")
