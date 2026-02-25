"""
Visualization Script for HFGM / CS Global Macro Analysis
Generates all charts for the web report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import statsmodels.api as sm
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Professional color scheme
COLORS = {
    'primary': '#1a3a5c',
    'secondary': '#2e86ab',
    'accent': '#e84855',
    'gold': '#f4a261',
    'green': '#2a9d8f',
    'light': '#f8f9fa',
    'gray': '#6c757d',
    'dark': '#212529',
    'bg': '#0d1b2a',
    'grid': '#e9ecef'
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
})

os.makedirs('/home/ubuntu/hfgm_analysis/charts', exist_ok=True)

# Load results
with open('/home/ubuntu/hfgm_analysis/data/analysis_results.pkl', 'rb') as f:
    results = pickle.load(f)

cs_backtest = results['cs_backtest']
hfgm_live = results['hfgm_live']
ff5_aligned = results['ff5_aligned']
bench_aligned = results['bench_aligned']
corr_matrix = results['corr_matrix']
backtest_metrics = results['backtest_metrics']
live_metrics = results['live_metrics']
eq_factor = results['eq_factor']
dur_factor = results['dur_factor']
trend_factor = results['trend_factor']
comm_factor = results['comm_factor']
carry_factor = results['carry_factor']
rf = results['rf']
model_5f = results['model_5f']
ff5_model = results['ff5_model']
predicted_5f = results['predicted_5f']
predicted_ff5 = results['predicted_ff5']
tracking_error_5f = results['tracking_error_5f']
tracking_error_ff5 = results['tracking_error_ff5']
ff5_results = results['ff5_results']

# ============================================================
# CHART 1: Cumulative Returns Comparison
# ============================================================
print("Generating Chart 1: Cumulative Returns...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1 = axes[0]
ax2 = axes[1]

# Cumulative returns
cum_cs = (1 + cs_backtest).cumprod()
cum_spy = (1 + bench_aligned['SPY']).cumprod()
cum_tlt = (1 + bench_aligned['TLT']).cumprod()
cum_60_40 = (1 + 0.6 * bench_aligned['SPY'] + 0.4 * bench_aligned['TLT']).cumprod()

# Normalize to 100
cum_cs = cum_cs / cum_cs.iloc[0] * 100
cum_spy = cum_spy / cum_spy.iloc[0] * 100
cum_tlt = cum_tlt / cum_tlt.iloc[0] * 100
cum_60_40 = cum_60_40 / cum_60_40.iloc[0] * 100

ax1.plot(cum_cs.index, cum_cs.values, color=COLORS['primary'], linewidth=2.5, 
         label='CS GM 2x Vol (Backtest)', zorder=5)
ax1.plot(cum_spy.index, cum_spy.values, color=COLORS['secondary'], linewidth=1.5, 
         alpha=0.8, label='S&P 500 (SPY)', linestyle='--')
ax1.plot(cum_tlt.index, cum_tlt.values, color=COLORS['green'], linewidth=1.5, 
         alpha=0.8, label='20yr Treasury (TLT)', linestyle='-.')
ax1.plot(cum_60_40.index, cum_60_40.values, color=COLORS['gold'], linewidth=1.5, 
         alpha=0.8, label='60/40 Portfolio', linestyle=':')

# Add HFGM live data
if not hfgm_live.empty:
    cum_hfgm = (1 + hfgm_live).cumprod()
    cum_hfgm = cum_hfgm / cum_hfgm.iloc[0] * 100
    ax1.plot(cum_hfgm.index, cum_hfgm.values, color=COLORS['accent'], linewidth=2.5, 
             label='HFGM ETF (Live)', zorder=6, marker='o', markersize=4)

ax1.set_title('Cumulative Performance: CS Global Macro Index vs Benchmarks\n(Indexed to 100, Jan 2000 – Dec 2025)', 
              fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)
ax1.set_ylabel('Cumulative Return (Indexed)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0f}'))

# Drawdown chart
cum_cs_raw = (1 + cs_backtest).cumprod()
rolling_max = cum_cs_raw.cummax()
drawdown = (cum_cs_raw - rolling_max) / rolling_max * 100

ax2.fill_between(drawdown.index, drawdown.values, 0, 
                  color=COLORS['accent'], alpha=0.4, label='CS GM Drawdown')
ax2.plot(drawdown.index, drawdown.values, color=COLORS['accent'], linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.set_xlabel('Date', fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))
ax2.legend(fontsize=9)

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart1_cumulative_returns.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 1 saved.")

# ============================================================
# CHART 2: FF5 Factor Regression Results
# ============================================================
print("Generating Chart 2: FF5 Regression Results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Factor loadings bar chart
ax1 = axes[0]
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
betas = [ff5_model.params[f] for f in factors]
conf_int = ff5_model.conf_int()
errors = [(conf_int.loc[f, 1] - conf_int.loc[f, 0]) / 2 for f in factors]
colors = [COLORS['green'] if b > 0 else COLORS['accent'] for b in betas]

bars = ax1.bar(factors, betas, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax1.errorbar(factors, betas, yerr=errors, fmt='none', color=COLORS['dark'], 
             capsize=5, linewidth=2, capthick=2)
ax1.axhline(y=0, color=COLORS['dark'], linewidth=1, linestyle='-')
ax1.set_title(f'FF5 Factor Loadings\n(R² = {ff5_model.rsquared:.3f}, Alpha = {ff5_model.params["const"]*12:.2%} p.a.)', 
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Beta Coefficient', fontsize=11)
ax1.set_xlabel('Fama-French Factor', fontsize=11)

for bar, beta in zip(bars, betas):
    ax1.text(bar.get_x() + bar.get_width()/2, 
             beta + (0.02 if beta >= 0 else -0.04),
             f'{beta:.3f}', ha='center', va='bottom' if beta >= 0 else 'top',
             fontsize=9, fontweight='bold')

# Model R-squared comparison
ax2 = axes[1]
models = ['FF5', 'FF6\n(FF5+MOM)', '3-Factor\nMacro', '4-Factor\nMacro', '5-Factor\nMacro']
r_squareds = [
    results['ff5_model'].rsquared,
    results['ff6_model'].rsquared if 'ff6_model' in results else 0.134,
    results['model_3f'].rsquared,
    results['model_4f'].rsquared,
    results['model_5f'].rsquared,
]
adj_r_squareds = [
    results['ff5_model'].rsquared_adj,
    results['ff6_model'].rsquared_adj if 'ff6_model' in results else 0.117,
    results['model_3f'].rsquared_adj,
    results['model_4f'].rsquared_adj,
    results['model_5f'].rsquared_adj,
]

x = np.arange(len(models))
width = 0.35
bars1 = ax2.bar(x - width/2, r_squareds, width, label='R²', 
                color=COLORS['primary'], alpha=0.8, edgecolor='white')
bars2 = ax2.bar(x + width/2, adj_r_squareds, width, label='Adj. R²', 
                color=COLORS['secondary'], alpha=0.8, edgecolor='white')

ax2.set_title('Model Comparison: R² by Factor Model\n(Higher = Better Explanatory Power)', 
              fontsize=12, fontweight='bold')
ax2.set_ylabel('R-Squared', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.2f}'))

for bar, val in zip(bars1, r_squareds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart2_ff5_regression.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 2 saved.")

# ============================================================
# CHART 3: 5-Factor Macro Model - Factor Loadings & Fit
# ============================================================
print("Generating Chart 3: 5-Factor Macro Model...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5-factor loadings
ax1 = axes[0]
macro_factors = ['Equity', 'Duration', 'Trend', 'Commodity', 'Carry']
macro_betas = [model_5f.params[f] for f in macro_factors]
macro_tstats = [model_5f.tvalues[f] for f in macro_factors]
macro_conf = model_5f.conf_int()
macro_errors = [(macro_conf.loc[f, 1] - macro_conf.loc[f, 0]) / 2 for f in macro_factors]

# Color by significance
sig_colors = []
for t in macro_tstats:
    if abs(t) > 2.58:
        sig_colors.append(COLORS['primary'])
    elif abs(t) > 1.96:
        sig_colors.append(COLORS['secondary'])
    else:
        sig_colors.append(COLORS['gray'])

bars = ax1.bar(macro_factors, macro_betas, color=sig_colors, alpha=0.85, 
               edgecolor='white', linewidth=1.5)
ax1.errorbar(macro_factors, macro_betas, yerr=macro_errors, fmt='none', 
             color=COLORS['dark'], capsize=5, linewidth=2, capthick=2)
ax1.axhline(y=0, color=COLORS['dark'], linewidth=1)
ax1.set_title(f'5-Factor Macro Model Loadings\n(R² = {model_5f.rsquared:.3f}, Alpha = {model_5f.params["const"]*12:.2%} p.a.)', 
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Beta Coefficient', fontsize=11)

for bar, beta, t in zip(bars, macro_betas, macro_tstats):
    stars = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
    ax1.text(bar.get_x() + bar.get_width()/2, 
             beta + (0.05 if beta >= 0 else -0.1),
             f'{beta:.3f}{stars}', ha='center', va='bottom' if beta >= 0 else 'top',
             fontsize=9, fontweight='bold')

# Legend for significance
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['primary'], label='p < 0.01 (***)'),
    Patch(facecolor=COLORS['secondary'], label='p < 0.05 (**)'),
    Patch(facecolor=COLORS['gray'], label='Not significant'),
]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper right')

# Actual vs Predicted scatter
ax2 = axes[1]
actual = cs_backtest.reindex(model_5f.fittedvalues.index)
predicted = predicted_5f

ax2.scatter(predicted.values, actual.values, alpha=0.4, color=COLORS['primary'], 
            s=20, edgecolors='none')
# Add regression line
z = np.polyfit(predicted.dropna().values, actual.reindex(predicted.dropna().index).values, 1)
p = np.poly1d(z)
x_line = np.linspace(predicted.min(), predicted.max(), 100)
ax2.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=2, label=f'Fit line')
ax2.plot([predicted.min(), predicted.max()], [predicted.min(), predicted.max()], 
         'k--', linewidth=1, alpha=0.5, label='45° line')
ax2.set_xlabel('Predicted Returns (5-Factor Macro Model)', fontsize=10)
ax2.set_ylabel('Actual CS GM Returns', fontsize=10)
ax2.set_title(f'Actual vs Predicted Returns\n(5-Factor Macro, R² = {model_5f.rsquared:.3f})', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart3_macro_model.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 3 saved.")

# ============================================================
# Helper function for performance metrics
# ============================================================
def compute_metrics_for_chart(returns, rf_series):
    rf_aligned = rf_series.reindex(returns.index).fillna(rf_series.mean())
    ann_ret = returns.mean() * 12 * 100
    ann_vol = returns.std() * np.sqrt(12) * 100
    sharpe = (returns.mean() - rf_aligned.mean()) / returns.std() * np.sqrt(12)
    cum_ret = (1 + returns).cumprod()
    rolling_max = cum_ret.cummax()
    max_dd = ((cum_ret - rolling_max) / rolling_max).min() * 100
    return [ann_ret, ann_vol, sharpe, max_dd]

# ============================================================
# CHART 4: Risk Metrics Dashboard
# ============================================================
print("Generating Chart 4: Risk Metrics Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a: Monthly Returns Distribution
ax1 = axes[0, 0]
monthly_rets = cs_backtest.values * 100
ax1.hist(monthly_rets, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white', linewidth=0.5)
ax1.axvline(x=np.mean(monthly_rets), color=COLORS['accent'], linewidth=2, 
            linestyle='--', label=f'Mean: {np.mean(monthly_rets):.2f}%')
ax1.axvline(x=np.percentile(monthly_rets, 5), color=COLORS['gold'], linewidth=2, 
            linestyle=':', label=f'VaR 95%: {np.percentile(monthly_rets, 5):.2f}%')
ax1.set_title('Monthly Returns Distribution\n(CS GM 2x Vol, 2000-2025)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Monthly Return (%)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend(fontsize=9)

# Add normal distribution overlay
x_range = np.linspace(monthly_rets.min(), monthly_rets.max(), 100)
mu, sigma = np.mean(monthly_rets), np.std(monthly_rets)
normal_pdf = stats.norm.pdf(x_range, mu, sigma) * len(monthly_rets) * (monthly_rets.max() - monthly_rets.min()) / 40
ax1.plot(x_range, normal_pdf, color=COLORS['secondary'], linewidth=2, 
         linestyle='-', alpha=0.8, label='Normal dist.')
ax1.legend(fontsize=8)

# 4b: Rolling Sharpe Ratio
ax2 = axes[0, 1]
rolling_ret = cs_backtest.rolling(24).mean() * 12
rolling_vol = cs_backtest.rolling(24).std() * np.sqrt(12)
rolling_rf = rf.rolling(24).mean() * 12
rolling_sharpe = (rolling_ret - rolling_rf) / rolling_vol

ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLORS['primary'], linewidth=2)
ax2.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                  where=rolling_sharpe.values > 0, alpha=0.3, color=COLORS['green'])
ax2.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                  where=rolling_sharpe.values < 0, alpha=0.3, color=COLORS['accent'])
ax2.axhline(y=0, color=COLORS['dark'], linewidth=1)
ax2.axhline(y=1.0, color=COLORS['gold'], linewidth=1, linestyle='--', alpha=0.7, label='Sharpe = 1.0')
ax2.set_title('Rolling 24-Month Sharpe Ratio\n(CS GM 2x Vol)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio', fontsize=10)
ax2.legend(fontsize=9)

# 4c: Factor Correlation Heatmap
ax3 = axes[1, 0]
corr_subset = corr_matrix.loc[
    ['CS_GM', 'Equity', 'Duration', 'Trend', 'Commodity', 'Carry', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
    ['CS_GM', 'Equity', 'Duration', 'Trend', 'Commodity', 'Carry', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
]

im = ax3.imshow(corr_subset.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(len(corr_subset.columns)))
ax3.set_yticks(range(len(corr_subset.index)))
ax3.set_xticklabels(corr_subset.columns, rotation=45, ha='right', fontsize=8)
ax3.set_yticklabels(corr_subset.index, fontsize=8)
ax3.set_title('Factor Correlation Matrix\n(Monthly Returns, 2000-2025)', fontsize=12, fontweight='bold')

for i in range(len(corr_subset.index)):
    for j in range(len(corr_subset.columns)):
        val = corr_subset.values[i, j]
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center', 
                fontsize=6, color='black' if abs(val) < 0.7 else 'white')

plt.colorbar(im, ax=ax3, shrink=0.8)

# 4d: Performance Metrics Comparison
ax4 = axes[1, 1]
metrics_names = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Max DD']
cs_vals = [
    backtest_metrics['Ann. Return'] * 100,
    backtest_metrics['Ann. Volatility'] * 100,
    backtest_metrics['Sharpe Ratio'],
    backtest_metrics['Max Drawdown'] * 100
]
spy_metrics = compute_metrics_for_chart(bench_aligned['SPY'], rf)
tlt_metrics = compute_metrics_for_chart(bench_aligned['TLT'], rf)

x = np.arange(len(metrics_names))
width = 0.25
ax4.bar(x - width, cs_vals, width, label='CS GM 2x Vol', color=COLORS['primary'], alpha=0.85)
ax4.bar(x, spy_metrics, width, label='S&P 500', color=COLORS['secondary'], alpha=0.85)
ax4.bar(x + width, tlt_metrics, width, label='20yr Treasury', color=COLORS['green'], alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_names, fontsize=10)
ax4.set_title('Performance Metrics Comparison\n(Backtest Period 2000-2025)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.axhline(y=0, color=COLORS['dark'], linewidth=0.5)

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart4_risk_metrics.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 4 saved.")

# ============================================================
# CHART 5: Live vs Backtest Comparison
# ============================================================
print("Generating Chart 5: Live vs Backtest Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Monthly returns comparison
ax1 = axes[0]
live_period = hfgm_live.index
bt_live = cs_backtest.reindex(live_period)

x = np.arange(len(live_period))
width = 0.35
ax1.bar(x - width/2, hfgm_live.values * 100, width, label='HFGM Live', 
        color=COLORS['accent'], alpha=0.85, edgecolor='white')
ax1.bar(x + width/2, bt_live.values * 100, width, label='CS GM Backtest', 
        color=COLORS['primary'], alpha=0.85, edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels([d.strftime('%b\n%Y') for d in live_period], fontsize=7, rotation=0)
ax1.set_title('Monthly Returns: HFGM Live vs CS GM Backtest\n(May 2025 – Feb 2026)', 
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Monthly Return (%)', fontsize=10)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color=COLORS['dark'], linewidth=0.5)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))

# Cumulative returns comparison
ax2 = axes[1]
cum_live = (1 + hfgm_live).cumprod()
cum_bt_live = (1 + bt_live).cumprod()

ax2.plot(live_period, cum_live.values * 100 - 100, color=COLORS['accent'], 
         linewidth=2.5, marker='o', markersize=5, label='HFGM Live')
ax2.plot(live_period, cum_bt_live.values * 100 - 100, color=COLORS['primary'], 
         linewidth=2.5, marker='s', markersize=5, label='CS GM Backtest', linestyle='--')
ax2.fill_between(live_period, cum_live.values * 100 - 100, cum_bt_live.values * 100 - 100,
                  alpha=0.2, color=COLORS['gold'], label='Tracking Gap')
ax2.set_title(f'Cumulative Returns: Live vs Backtest\n(Corr = {hfgm_live.corr(bt_live):.3f})', 
              fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
ax2.legend(fontsize=10)
ax2.axhline(y=0, color=COLORS['dark'], linewidth=0.5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart5_live_vs_backtest.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 5 saved.")

# ============================================================
# CHART 6: Risk Premia Decomposition
# ============================================================
print("Generating Chart 6: Risk Premia Decomposition...")

fig, ax = plt.subplots(figsize=(12, 7))

# Rolling 36-month contribution of each factor
window = 36
factor_data = pd.concat([
    eq_factor.rename('Equity Risk'),
    dur_factor.rename('Duration/Bond'),
    trend_factor.rename('Trend Following'),
    comm_factor.rename('Commodity/Inflation'),
    carry_factor.rename('Currency Carry'),
], axis=1).reindex(cs_backtest.index).dropna()

# Estimate rolling factor contributions
factor_colors = [COLORS['secondary'], COLORS['green'], COLORS['primary'], 
                 COLORS['gold'], COLORS['accent']]

# Use overall betas for decomposition
betas_5f = {
    'Equity Risk': model_5f.params['Equity'],
    'Duration/Bond': model_5f.params['Duration'],
    'Trend Following': model_5f.params['Trend'],
    'Commodity/Inflation': model_5f.params['Commodity'],
    'Currency Carry': model_5f.params['Carry'],
}

contributions = pd.DataFrame()
for factor, beta in betas_5f.items():
    contributions[factor] = factor_data[factor] * beta

# Rolling 12-month average contributions
rolling_contrib = contributions.rolling(12).mean() * 12 * 100  # Annualized %

# Stack plot
ax.stackplot(rolling_contrib.index, 
             [rolling_contrib[f].values for f in rolling_contrib.columns],
             labels=list(rolling_contrib.columns),
             colors=factor_colors, alpha=0.75)

# Overlay actual returns
rolling_actual = cs_backtest.rolling(12).mean() * 12 * 100
ax.plot(rolling_actual.index, rolling_actual.values, 'k-', linewidth=2.5, 
        label='Actual Return (12M Rolling)', zorder=10)

ax.set_title('Risk Premia Decomposition: Factor Return Contributions\n(CS GM 2x Vol, Rolling 12-Month Annualized)', 
             fontsize=13, fontweight='bold')
ax.set_ylabel('Annualized Return Contribution (%)', fontsize=11)
ax.set_xlabel('Date', fontsize=10)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))

plt.tight_layout(pad=2)
plt.savefig('/home/ubuntu/hfgm_analysis/charts/chart6_risk_premia.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Chart 6 saved.")

print("\nAll charts generated successfully!")
