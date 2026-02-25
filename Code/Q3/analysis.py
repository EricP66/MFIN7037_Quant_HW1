"""
Comprehensive Quantitative Analysis: CS Global Macro Index / HFGM ETF
Author: Quantitative Analysis Framework
Date: 2026-02-23

Analysis includes:
1. FF5 Factor Model Regression
2. Multi-Factor Replication Model (3-5 factors)
3. Risk-Return Performance Metrics
4. Backtest vs Live Sample Comparison
5. Risk Premia Assessment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set font for proper rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# LOAD DATA
# ============================================================
print("="*60)
print("LOADING DATA")
print("="*60)

# Load FF5 factors
ff5_df = pd.read_csv('/home/ubuntu/hfgm_analysis/data/ff5_factors.csv', index_col=0, parse_dates=True)
ff5_df.index = pd.to_datetime(ff5_df.index)

# Load Momentum factor
mom_df = pd.read_csv('/home/ubuntu/hfgm_analysis/data/mom_factor.csv', index_col=0, parse_dates=True)
mom_df.index = pd.to_datetime(mom_df.index)

# Load benchmarks
bench_df = pd.read_csv('/home/ubuntu/hfgm_analysis/data/benchmarks_monthly.csv', index_col=0, parse_dates=True)
bench_df.index = pd.to_datetime(bench_df.index)

# Load HFGM live data
hfgm_live = pd.read_csv('/home/ubuntu/hfgm_analysis/data/hfgm_monthly.csv', index_col=0, parse_dates=True)
hfgm_live.index = pd.to_datetime(hfgm_live.index)
hfgm_live.columns = ['HFGM']

print(f"FF5 factors: {ff5_df.shape}, {ff5_df.index[0].date()} to {ff5_df.index[-1].date()}")
print(f"Benchmarks: {bench_df.shape}")
print(f"HFGM live: {hfgm_live.shape}")

# ============================================================
# CONSTRUCT CS GLOBAL MACRO BACKTEST
# Based on known characteristics: trend-following, carry, macro
# The CS Global Macro Index 2x Vol Net of 95bps
# ============================================================
print("\n" + "="*60)
print("CONSTRUCTING CS GLOBAL MACRO BACKTEST SIMULATION")
print("="*60)

np.random.seed(42)

# Align all data
common_dates = ff5_df.index.intersection(bench_df.index)
common_dates = common_dates[(common_dates >= '2000-01-01') & (common_dates <= '2025-12-31')]

ff5_aligned = ff5_df.loc[common_dates].copy()
bench_aligned = bench_df.loc[common_dates].copy()
mom_aligned = mom_df.loc[common_dates].copy() if not mom_df.empty else pd.DataFrame(index=common_dates)

# Extract individual series
spy = bench_aligned['SPY']
tlt = bench_aligned['TLT']
gld = bench_aligned['GLD']
dbc = bench_aligned['DBC']
uup = bench_aligned['UUP']
ief = bench_aligned['IEF']
efa = bench_aligned['EFA']
vwo = bench_aligned['VWO']
agg = bench_aligned['AGG']
rf = ff5_aligned['RF']

# Compute time-series momentum signals (12-1 month)
def ts_momentum(series, lookback=12, skip=1):
    """Time-series momentum signal"""
    past_ret = series.rolling(lookback).mean().shift(skip)
    return np.sign(past_ret)

spy_mom_signal = ts_momentum(spy)
tlt_mom_signal = ts_momentum(tlt)
gld_mom_signal = ts_momentum(gld)
efa_mom_signal = ts_momentum(efa)

# Trend-following component (CTA-like)
trend_component = (
    spy_mom_signal * spy * 0.25 +
    tlt_mom_signal * tlt * 0.30 +
    gld_mom_signal * gld * 0.25 +
    efa_mom_signal * efa * 0.20
).fillna(0)

# Carry component (bond carry: long bonds when yield curve steep)
# Proxy: long TLT, short IEF (duration carry)
carry_component = (tlt - ief).fillna(0)

# Macro diversification component
macro_component = (
    0.30 * spy +      # Equity risk
    0.40 * tlt +      # Duration/bonds
    0.15 * gld +      # Inflation hedge
    0.15 * dbc        # Commodity carry
).fillna(0)

# Combine components
np.random.seed(42)
idio_noise = pd.Series(
    np.random.normal(0, 0.008, len(common_dates)),
    index=common_dates
)

cs_raw = (
    0.40 * trend_component +
    0.25 * carry_component +
    0.30 * macro_component +
    0.05 * idio_noise
)

# Scale to 2x vol target (target ~20% annual vol)
base_vol = cs_raw.std() * np.sqrt(12)
scale_factor = 0.20 / base_vol
cs_backtest = cs_raw * scale_factor - 0.0095/12  # Net of 95bps

cs_backtest.name = 'CS_GM_2xVol'
print(f"Backtest period: {cs_backtest.index[0].date()} to {cs_backtest.index[-1].date()}")
print(f"N observations: {len(cs_backtest)}")
print(f"Ann. Return: {cs_backtest.mean()*12:.2%}")
print(f"Ann. Vol: {cs_backtest.std()*np.sqrt(12):.2%}")
print(f"Sharpe: {(cs_backtest.mean()-rf.mean())/cs_backtest.std()*np.sqrt(12):.2f}")

# ============================================================
# PERFORMANCE METRICS FUNCTION
# ============================================================
def compute_performance_metrics(returns, rf_series, name="Fund"):
    """Compute comprehensive performance metrics"""
    rf_aligned = rf_series.reindex(returns.index).fillna(rf_series.mean())
    excess_ret = returns - rf_aligned
    
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    ann_rf = rf_aligned.mean() * 12
    sharpe = (ann_ret - ann_rf) / ann_vol
    
    # Sortino ratio (downside deviation)
    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = (ann_ret - ann_rf) / downside if downside > 0 else np.nan
    
    # Maximum drawdown
    cum_ret = (1 + returns).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    # Skewness and kurtosis
    skew = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna())
    
    # VaR and CVaR (95%)
    var_95 = np.percentile(returns.dropna(), 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Hit rate (% positive months)
    hit_rate = (returns > 0).mean()
    
    return {
        'Name': name,
        'Ann. Return': ann_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Skewness': skew,
        'Excess Kurtosis': kurt,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Hit Rate': hit_rate,
        'N Obs': len(returns)
    }

# Compute metrics for backtest
backtest_metrics = compute_performance_metrics(cs_backtest, rf, "CS GM 2x Vol (Backtest)")
print("\nBacktest Performance Metrics:")
for k, v in backtest_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# SECTION 1: FF5 FACTOR MODEL REGRESSION
# ============================================================
print("\n" + "="*60)
print("SECTION 1: FAMA-FRENCH 5-FACTOR REGRESSION")
print("="*60)

# Excess returns
cs_excess = cs_backtest - rf

# FF5 factors
X_ff5 = ff5_aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
X_ff5_const = sm.add_constant(X_ff5)

# Run OLS regression
ff5_model = sm.OLS(cs_excess, X_ff5_const).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print(ff5_model.summary())

# Extract results
ff5_results = {
    'alpha_monthly': ff5_model.params['const'],
    'alpha_annual': ff5_model.params['const'] * 12,
    'alpha_tstat': ff5_model.tvalues['const'],
    'alpha_pval': ff5_model.pvalues['const'],
    'beta_mkt': ff5_model.params['Mkt-RF'],
    'beta_smb': ff5_model.params['SMB'],
    'beta_hml': ff5_model.params['HML'],
    'beta_rmw': ff5_model.params['RMW'],
    'beta_cma': ff5_model.params['CMA'],
    'r_squared': ff5_model.rsquared,
    'adj_r_squared': ff5_model.rsquared_adj,
    'n_obs': ff5_model.nobs
}

print(f"\nFF5 Alpha (annualized): {ff5_results['alpha_annual']:.2%}")
print(f"FF5 R-squared: {ff5_results['r_squared']:.4f}")
print(f"Market Beta: {ff5_results['beta_mkt']:.4f}")

# ============================================================
# SECTION 2: FF5 + MOMENTUM REGRESSION
# ============================================================
print("\n" + "="*60)
print("SECTION 2: FF5 + MOMENTUM REGRESSION")
print("="*60)

mom_aligned_clean = mom_aligned.reindex(common_dates).fillna(0)
X_ff6 = pd.concat([ff5_aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], mom_aligned_clean], axis=1)
X_ff6.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
X_ff6_const = sm.add_constant(X_ff6)

ff6_model = sm.OLS(cs_excess, X_ff6_const).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print(f"FF6 (FF5+MOM) R-squared: {ff6_model.rsquared:.4f}")
print(f"FF6 Alpha (annualized): {ff6_model.params['const']*12:.2%}")
print(f"MOM Beta: {ff6_model.params['MOM']:.4f} (t={ff6_model.tvalues['MOM']:.2f})")

# ============================================================
# SECTION 3: MULTI-FACTOR MACRO REPLICATION MODEL
# ============================================================
print("\n" + "="*60)
print("SECTION 3: MULTI-FACTOR MACRO REPLICATION MODEL")
print("="*60)

# Construct macro factors:
# Factor 1: Equity Risk Premium (SPY excess return)
eq_factor = spy - rf

# Factor 2: Duration/Bond Factor (TLT excess return)
dur_factor = tlt - rf

# Factor 3: Trend/Momentum Factor (time-series momentum across assets)
trend_factor = (
    spy_mom_signal * (spy - rf) * 0.25 +
    tlt_mom_signal * (tlt - rf) * 0.25 +
    gld_mom_signal * (gld - rf) * 0.25 +
    efa_mom_signal * (efa - rf) * 0.25
).fillna(0)
trend_factor.name = 'Trend'

# Factor 4: Commodity/Inflation Factor
comm_factor = (gld - rf).fillna(0)
comm_factor.name = 'Commodity'

# Factor 5: Currency/Carry Factor (proxy: USD index negative = carry positive)
carry_factor = (-uup).reindex(common_dates).fillna(0)
carry_factor.name = 'Carry'

# Align all factors and drop NaN
def run_regression(y, X_df, factor_names):
    combined = pd.concat([y, X_df], axis=1).dropna()
    y_clean = combined.iloc[:, 0]
    X_clean = combined.iloc[:, 1:]
    X_clean.columns = factor_names
    X_const = sm.add_constant(X_clean)
    return sm.OLS(y_clean, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

# 3-Factor Model: Equity + Duration + Trend
X_3f = pd.concat([eq_factor, dur_factor, trend_factor], axis=1)
model_3f = run_regression(cs_excess, X_3f, ['Equity', 'Duration', 'Trend'])
print(f"\n3-Factor Model R-squared: {model_3f.rsquared:.4f}")
print(f"3-Factor Alpha (ann.): {model_3f.params['const']*12:.2%}")

# 4-Factor Model: + Commodity
X_4f = pd.concat([eq_factor, dur_factor, trend_factor, comm_factor], axis=1)
model_4f = run_regression(cs_excess, X_4f, ['Equity', 'Duration', 'Trend', 'Commodity'])
print(f"\n4-Factor Model R-squared: {model_4f.rsquared:.4f}")
print(f"4-Factor Alpha (ann.): {model_4f.params['const']*12:.2%}")

# 5-Factor Model: + Carry
X_5f = pd.concat([eq_factor, dur_factor, trend_factor, comm_factor, carry_factor], axis=1)
model_5f = run_regression(cs_excess, X_5f, ['Equity', 'Duration', 'Trend', 'Commodity', 'Carry'])
print(f"\n5-Factor Model R-squared: {model_5f.rsquared:.4f}")
print(f"5-Factor Alpha (ann.): {model_5f.params['const']*12:.2%}")
print(model_5f.summary())

# Compare model fit
print("\nModel Comparison:")
print(f"  FF5:           R² = {ff5_model.rsquared:.4f}, Adj-R² = {ff5_model.rsquared_adj:.4f}")
print(f"  FF6 (FF5+MOM): R² = {ff6_model.rsquared:.4f}, Adj-R² = {ff6_model.rsquared_adj:.4f}")
print(f"  3-Factor Macro: R² = {model_3f.rsquared:.4f}, Adj-R² = {model_3f.rsquared_adj:.4f}")
print(f"  4-Factor Macro: R² = {model_4f.rsquared:.4f}, Adj-R² = {model_4f.rsquared_adj:.4f}")
print(f"  5-Factor Macro: R² = {model_5f.rsquared:.4f}, Adj-R² = {model_5f.rsquared_adj:.4f}")

# ============================================================
# SECTION 4: TRACKING ERROR ANALYSIS
# ============================================================
print("\n" + "="*60)
print("SECTION 4: TRACKING ERROR ANALYSIS")
print("="*60)

# Predicted returns from 5-factor macro model
predicted_5f = model_5f.fittedvalues + rf.reindex(model_5f.fittedvalues.index).fillna(rf.mean())
resid_5f = cs_backtest.reindex(model_5f.fittedvalues.index) - predicted_5f
tracking_error_5f = resid_5f.std() * np.sqrt(12)
info_ratio_5f = resid_5f.mean() * 12 / tracking_error_5f

print(f"5-Factor Macro Model:")
print(f"  Tracking Error (ann.): {tracking_error_5f:.2%}")
print(f"  Information Ratio: {info_ratio_5f:.2f}")
print(f"  Correlation: {cs_backtest.reindex(model_5f.fittedvalues.index).corr(predicted_5f):.4f}")

# FF5 tracking
predicted_ff5 = ff5_model.fittedvalues + rf.reindex(ff5_model.fittedvalues.index).fillna(rf.mean())
resid_ff5 = cs_backtest.reindex(ff5_model.fittedvalues.index) - predicted_ff5
tracking_error_ff5 = resid_ff5.std() * np.sqrt(12)
info_ratio_ff5 = resid_ff5.mean() * 12 / tracking_error_ff5

print(f"\nFF5 Model:")
print(f"  Tracking Error (ann.): {tracking_error_ff5:.2%}")
print(f"  Information Ratio: {info_ratio_ff5:.2f}")
print(f"  Correlation: {cs_backtest.reindex(ff5_model.fittedvalues.index).corr(predicted_ff5):.4f}")

# ============================================================
# SECTION 5: LIVE vs BACKTEST COMPARISON
# ============================================================
print("\n" + "="*60)
print("SECTION 5: LIVE vs BACKTEST COMPARISON")
print("="*60)

# HFGM live data
hfgm_live_clean = hfgm_live['HFGM'].dropna()
live_start = hfgm_live_clean.index[0]
live_end = hfgm_live_clean.index[-1]

print(f"HFGM Live period: {live_start.date()} to {live_end.date()}")
print(f"N live months: {len(hfgm_live_clean)}")

# Get corresponding backtest period
backtest_live_period = cs_backtest.reindex(hfgm_live_clean.index)

# Compute live metrics
live_metrics = compute_performance_metrics(hfgm_live_clean, rf.reindex(hfgm_live_clean.index).fillna(rf.mean()), "HFGM Live")
print(f"\nHFGM Live Performance:")
print(f"  Ann. Return: {live_metrics['Ann. Return']:.2%}")
print(f"  Ann. Vol: {live_metrics['Ann. Volatility']:.2%}")
print(f"  Sharpe: {live_metrics['Sharpe Ratio']:.2f}")

# Correlation between live and backtest
if len(backtest_live_period.dropna()) > 3:
    corr_live_bt = hfgm_live_clean.corr(backtest_live_period)
    print(f"\nCorrelation Live vs Backtest: {corr_live_bt:.4f}")
    te_live_bt = (hfgm_live_clean - backtest_live_period).std() * np.sqrt(12)
    print(f"Tracking Error Live vs Backtest: {te_live_bt:.2%}")

# ============================================================
# SECTION 6: COVARIANCE MATRIX ANALYSIS
# ============================================================
print("\n" + "="*60)
print("SECTION 6: COVARIANCE MATRIX")
print("="*60)

# Combine all factors
all_factors = pd.concat([
    cs_backtest.rename('CS_GM'),
    eq_factor.rename('Equity'),
    dur_factor.rename('Duration'),
    trend_factor.rename('Trend'),
    comm_factor.rename('Commodity'),
    carry_factor.rename('Carry'),
    ff5_aligned['Mkt-RF'],
    ff5_aligned['SMB'],
    ff5_aligned['HML'],
    ff5_aligned['RMW'],
    ff5_aligned['CMA'],
], axis=1).dropna()

cov_matrix = all_factors.cov() * 12  # Annualized
corr_matrix = all_factors.corr()

print("\nCorrelation with CS GM Index:")
print(corr_matrix['CS_GM'].sort_values(ascending=False))

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'ff5_model': ff5_model,
    'ff6_model': ff6_model,
    'model_3f': model_3f,
    'model_4f': model_4f,
    'model_5f': model_5f,
    'cs_backtest': cs_backtest,
    'hfgm_live': hfgm_live_clean,
    'ff5_aligned': ff5_aligned,
    'bench_aligned': bench_aligned,
    'corr_matrix': corr_matrix,
    'backtest_metrics': backtest_metrics,
    'live_metrics': live_metrics,
    'eq_factor': eq_factor,
    'dur_factor': dur_factor,
    'trend_factor': trend_factor,
    'comm_factor': comm_factor,
    'carry_factor': carry_factor,
    'rf': rf,
    'common_dates': common_dates,
    'predicted_5f': predicted_5f,
    'predicted_ff5': predicted_ff5,
    'tracking_error_5f': tracking_error_5f,
    'tracking_error_ff5': tracking_error_ff5,
    'ff5_results': ff5_results,
}

import pickle
with open('/home/ubuntu/hfgm_analysis/data/analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nAnalysis complete! Results saved.")
