import pandas as pd
import os

DATA_DIR = r"c:\Users\hp\Desktop\New_ML\Moses-RandomForest\data"
nifty_daily = os.path.join(DATA_DIR, "nifty_daily.csv")
nifty_15m = os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv")

if os.path.exists(nifty_daily):
    df_d = pd.read_csv(nifty_daily)
    print(f"Daily latest: {df_d.iloc[-1]['date']}")
else:
    print("Daily missing")

if os.path.exists(nifty_15m):
    df_15m = pd.read_csv(nifty_15m)
    # Check both 'date' and 'datetime'
    date_col = 'date' if 'date' in df_15m.columns else 'datetime'
    print(f"15m latest: {df_15m.iloc[-1][date_col]}")
else:
    print("15m missing")
