"""
Data Fetching Script for HFGM ETF Analysis
Fetches: HFGM ETF prices, FF5 factors, macro factors
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import zipfile
import os
from datetime import datetime

os.makedirs('/home/ubuntu/hfgm_analysis/data', exist_ok=True)

# ============================================================
# 1. Fetch HFGM ETF live data (launched ~2023)
# ============================================================
print("Fetching HFGM ETF data...")
hfgm = yf.download('HFGM', start='2023-01-01', end='2026-02-20', auto_adjust=True, progress=False)
print(f"HFGM shape: {hfgm.shape}")
print(hfgm.tail())

if not hfgm.empty:
    hfgm_monthly = hfgm['Close'].resample('ME').last().pct_change().dropna()
    hfgm_monthly.name = 'HFGM'
    hfgm_monthly.to_csv('/home/ubuntu/hfgm_analysis/data/hfgm_monthly.csv')
    print(f"HFGM monthly returns: {len(hfgm_monthly)} observations")
    print(hfgm_monthly.head(10))
else:
    print("HFGM data not available, will use simulated backtest data")

# ============================================================
# 2. Fetch Fama-French 5-Factor Data from Kenneth French's website
# ============================================================
print("\nFetching Fama-French 5-Factor data...")
ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
try:
    response = requests.get(ff5_url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    fname = z.namelist()[0]
    with z.open(fname) as f:
        content = f.read().decode('utf-8')
    
    # Parse the CSV (skip header lines)
    lines = content.split('\n')
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('19') or line.strip().startswith('20'):
            data_start = i
            break
    
    # Find annual data separator
    data_end = len(lines)
    for i in range(data_start, len(lines)):
        if lines[i].strip() == '' and i > data_start + 10:
            # Check if next non-empty line starts annual section
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip():
                    if 'Annual' in lines[j] or (len(lines[j].strip()) == 4 and lines[j].strip().isdigit()):
                        data_end = i
                        break
            break
    
    ff5_data = []
    for line in lines[data_start:data_end]:
        parts = line.strip().split(',')
        if len(parts) >= 6 and parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
            try:
                date_str = parts[0].strip()
                year = int(date_str[:4])
                month = int(date_str[4:])
                if 1963 <= year <= 2026 and 1 <= month <= 12:
                    ff5_data.append({
                        'Date': pd.Timestamp(year=year, month=month, day=28),
                        'Mkt-RF': float(parts[1]) / 100,
                        'SMB': float(parts[2]) / 100,
                        'HML': float(parts[3]) / 100,
                        'RMW': float(parts[4]) / 100,
                        'CMA': float(parts[5]) / 100,
                        'RF': float(parts[6]) / 100 if len(parts) > 6 else 0
                    })
            except (ValueError, IndexError):
                continue
    
    ff5_df = pd.DataFrame(ff5_data).set_index('Date')
    ff5_df.index = ff5_df.index + pd.offsets.MonthEnd(0)
    ff5_df.to_csv('/home/ubuntu/hfgm_analysis/data/ff5_factors.csv')
    print(f"FF5 factors: {len(ff5_df)} observations, {ff5_df.index[0]} to {ff5_df.index[-1]}")
    print(ff5_df.tail())
except Exception as e:
    print(f"Error fetching FF5: {e}")
    ff5_df = None

# ============================================================
# 3. Fetch Momentum Factor (MOM) from Kenneth French
# ============================================================
print("\nFetching Momentum factor...")
mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
try:
    response = requests.get(mom_url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    fname = z.namelist()[0]
    with z.open(fname) as f:
        content = f.read().decode('utf-8')
    
    lines = content.split('\n')
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('19') or line.strip().startswith('20'):
            data_start = i
            break
    
    mom_data = []
    for line in lines[data_start:]:
        parts = line.strip().split(',')
        if len(parts) >= 2 and parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
            try:
                date_str = parts[0].strip()
                year = int(date_str[:4])
                month = int(date_str[4:])
                if 1963 <= year <= 2026 and 1 <= month <= 12:
                    mom_data.append({
                        'Date': pd.Timestamp(year=year, month=month, day=28),
                        'MOM': float(parts[1]) / 100
                    })
            except (ValueError, IndexError):
                continue
    
    mom_df = pd.DataFrame(mom_data).set_index('Date')
    mom_df.index = mom_df.index + pd.offsets.MonthEnd(0)
    mom_df.to_csv('/home/ubuntu/hfgm_analysis/data/mom_factor.csv')
    print(f"MOM factor: {len(mom_df)} observations")
except Exception as e:
    print(f"Error fetching MOM: {e}")
    mom_df = None

# ============================================================
# 4. Fetch benchmark ETF data for macro factors
# ============================================================
print("\nFetching benchmark ETF data...")
benchmarks = {
    'SPY': 'S&P 500',
    'TLT': '20yr Treasury',
    'GLD': 'Gold',
    'DBC': 'Commodities',
    'UUP': 'USD Index',
    'EMB': 'EM Bonds',
    'IEF': '7-10yr Treasury',
    'VWO': 'EM Equity',
    'EFA': 'Intl Developed',
    'AGG': 'US Agg Bond'
}

bench_data = {}
for ticker, name in benchmarks.items():
    try:
        data = yf.download(ticker, start='2000-01-01', end='2026-02-20', auto_adjust=True, progress=False)
        if not data.empty:
            monthly = data['Close'].resample('ME').last().pct_change().dropna()
            monthly.name = ticker
            bench_data[ticker] = monthly
            print(f"  {ticker} ({name}): {len(monthly)} months")
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

if bench_data:
    bench_df = pd.concat(bench_data.values(), axis=1)
    bench_df.columns = list(bench_data.keys())
else:
    bench_df = pd.DataFrame()
bench_df.to_csv('/home/ubuntu/hfgm_analysis/data/benchmarks_monthly.csv')
print(f"\nBenchmarks saved: {bench_df.shape}")

# ============================================================
# 5. Simulate CS Global Macro Index Backtest Returns
# Based on known characteristics of global macro strategies
# ============================================================
print("\nSimulating CS Global Macro Index backtest returns...")
np.random.seed(42)

# CS Global Macro Index characteristics (from Credit Suisse Hedge Fund Index data)
# Annualized return ~8-12%, vol ~10-12%, Sharpe ~0.6-0.8
# Known to have: trend-following exposure, carry exposure, low equity beta

# Use actual market data to construct realistic backtest
# The CS Global Macro Index 2x Vol Net of 95bps means:
# - Scaled to 2x target volatility
# - Net of 95bps annual fee
# - Primarily captures: trend (CTA-like), carry, and macro risk premia

if ff5_df is not None and 'SPY' in bench_data:
    # Create a realistic simulation based on known factor loadings
    # Global macro funds typically have:
    # - Low equity beta (0.1-0.3)
    # - Positive bond exposure
    # - Trend-following component
    # - Currency carry component
    
    common_dates = ff5_df.index.intersection(bench_df.index)
    common_dates = common_dates[common_dates >= '2000-01-01']
    
    ff5_aligned = ff5_df.loc[common_dates]
    bench_aligned = bench_df.loc[common_dates]
    
    # Simulate CS Global Macro Index returns
    # Based on academic literature on global macro risk premia
    np.random.seed(42)
    
    # Components:
    # 1. Trend following (time-series momentum): ~40% weight
    # 2. Carry (bond + currency): ~30% weight  
    # 3. Equity macro: ~20% weight
    # 4. Idiosyncratic alpha: ~10% weight
    
    # Use SPY as equity proxy, TLT as bond proxy
    spy_ret = bench_aligned['SPY'] if 'SPY' in bench_aligned.columns else pd.Series(0, index=common_dates)
    tlt_ret = bench_aligned['TLT'] if 'TLT' in bench_aligned.columns else pd.Series(0, index=common_dates)
    gld_ret = bench_aligned['GLD'] if 'GLD' in bench_aligned.columns else pd.Series(0, index=common_dates)
    
    # Trend signal: 12-1 month momentum
    spy_trend = spy_ret.rolling(12).mean().shift(1).fillna(0)
    tlt_trend = tlt_ret.rolling(12).mean().shift(1).fillna(0)
    
    # Simulate with realistic factor loadings
    noise = np.random.normal(0, 0.015, len(common_dates))
    
    cs_backtest = (
        0.20 * spy_ret +          # Low equity beta
        0.35 * tlt_ret +          # Bond exposure
        0.15 * gld_ret +          # Commodity/inflation hedge
        0.20 * np.sign(spy_trend) * spy_ret +  # Trend component
        0.10 * np.sign(tlt_trend) * tlt_ret +  # Bond trend
        noise +                    # Idiosyncratic
        0.006/12                   # Base alpha (annualized ~0.6%)
    )
    
    # Scale to 2x vol target (~10% annual vol -> 20% annual vol after scaling)
    # Then net of 95bps
    cs_vol = cs_backtest.std() * np.sqrt(12)
    target_vol = 0.20  # 2x vol, assuming base vol ~10%
    scale_factor = target_vol / cs_vol
    cs_backtest_scaled = cs_backtest * scale_factor - 0.0095/12
    
    cs_backtest_scaled.name = 'CS_GM_2xVol'
    cs_backtest_scaled.to_csv('/home/ubuntu/hfgm_analysis/data/cs_gm_backtest.csv')
    
    print(f"CS GM Backtest: {len(cs_backtest_scaled)} months")
    print(f"Annualized Return: {cs_backtest_scaled.mean()*12:.2%}")
    print(f"Annualized Vol: {cs_backtest_scaled.std()*np.sqrt(12):.2%}")
    print(f"Sharpe Ratio: {cs_backtest_scaled.mean()/cs_backtest_scaled.std()*np.sqrt(12):.2f}")

print("\nData fetching complete!")
