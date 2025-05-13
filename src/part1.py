# Part 1: Exploratory Data Analysis
# Import libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import zscore
import itertools

def prepare_data(cfg: dict) -> pd.DataFrame:
    """
    Loads raw price data, performs EDA (including weekend imputation,
    correlation analysis, and rolling correlations), saves all figures
    and tables to outputs, and returns the imputed log returns DataFrame.
    """
    # --- Load configuration ---
    raw_data_path = cfg["data"]["raw_data"]
    peak = pd.to_datetime(cfg["dates"]["peak"])
    crash_end = pd.to_datetime(cfg["dates"].get("crash_end", cfg["dates"]["trough"]))
    rolling_window = cfg["windows"].get("rolling_corr", 30)
    out_path = cfg["paths"]["log_returns_csv"]
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load
    data = pd.read_csv(raw_data_path, sep=',')
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    data.set_index('Date', inplace=True)

    # 2) Descriptive table
    summary = data.describe()
    summary.loc['skewness'] = data.skew()
    summary.loc['kurtosis'] = data.kurt()
    summary.to_csv(f"{out_dir}/descriptive_stats.csv")

    # 3) Normalized price plot
    normalized = data.div(data.iloc[0])
    plt.figure(figsize=(12,6))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col, color=sns.color_palette("husl", len(normalized.columns))[list(normalized.columns).index(col)])
    plt.axvspan(peak, crash_end, color='lightgray', alpha=0.3, label='Crash Period')
    plt.title("Normalized Asset Prices")
    plt.ylabel("Relative Price")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/normalized_prices.png")
    plt.close()

    # 4) Density plots of standardized prices
    std_data = data.apply(lambda x: (x - x.mean())/x.std(), axis=0)
    plt.figure(figsize=(12,6))
    for col in std_data.columns:
        sns.kdeplot(std_data[col], label=col, fill=False, alpha=0.7)
    plt.title("Standardized Density Plot")
    plt.xlabel("Standardized Prices")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/standardized_density.png")
    plt.close()

    # 5) Linear vs Log returns density
    lin_ret = ((data - data.shift(1))/data.shift(1)).dropna()
    log_ret = np.log(data/data.shift(1)).dropna()
    plt.figure(figsize=(12,6))
    sns.kdeplot(lin_ret.values.flatten(), color='blue', label='Linear Returns', linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(log_ret.values.flatten(), color='orange', label='Log Returns', linewidth=2, fill=True, alpha=0.3)
    plt.title("Density Plot: Linear vs Log Returns", fontsize=14, pad=20)
    plt.xlabel('Returns', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine()
    plt.axvline(x=np.mean(lin_ret), color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=np.mean(log_ret), color='orange', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/lin_vs_log_density.png")
    plt.close()

    # 6) Boxplot of log returns
    plt.figure(figsize=(10,8))
    sns.boxplot(data=log_ret, orient='h', fliersize=2, linewidth=1.2)
    plt.title("Boxplot of Log Returns", fontsize=14)
    plt.xlabel("Daily Log Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/boxplot_log_returns.png")
    plt.close()

    # 7) Monthly outliers plot
    z_score = zscore(log_ret)
    out_mask = z_score > 3  # Should it be z_score.abs() > 3?
    out_per_date = pd.DataFrame(out_mask, index=log_ret.index, columns=log_ret.columns).sum(axis=1)
    out_month = out_per_date.resample('ME').sum()

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(out_month.index, out_month.values, color='red', marker='o', linewidth=2)
    plt.axvspan(peak, crash_end, color='lightgray', alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Show every 2nd month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., "Jan 2022"
    plt.xticks(rotation=45)
    ax.set_title("Monthly Number of Outliers (>3σ)", fontsize=14) # Different from the original
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Outlier Count", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/monthly_outliers.png")
    plt.close()

    # 8) Weekend imputation & saving log_returns
    weekend = log_ret.index.weekday >= 5 # 5 for Saturday, 6 for Sunday

    plt.scatter(log_ret.index[weekend], log_ret['SOFR'][weekend], color='red', label='Weekend SOFR', zorder=5)
    plt.title('SOFR log return over weekend', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/log_return_weekend.png")
    plt.close()
    
    trad_assets = ['SPXT','XCMP','SOFR','VIX']
    for asset in trad_assets:
        log_ret[asset] = log_ret[asset].where(~weekend, None).ffill()

    # 9) Correlation heatmap (on imputed log returns)
    plt.figure(figsize=(12,8))
    sns.heatmap(log_ret.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0.2,
                vmin=-1, vmax=1, cbar_kws={'label':'Correlation'})
    plt.title("Correlation Heatmap")
    plt.savefig(f"{out_dir}/correlation_heatmap.png")
    plt.close()

    # 10) Rolling correlations grid
    window = rolling_window # days
    pairs = list(itertools.combinations(log_ret.columns, 2))
    n_cols = 4 # number of subplots per row
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows), sharex=True)
    axes = axes.flatten()
    for i, (c1, c2) in enumerate(pairs):
        rc = data[c1].rolling(window).corr(data[c2])
        axes[i].plot(rc.index, rc.values)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[i].set_title(f"{c1} vs {c2}")
        axes[i].grid(True)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(f"{out_dir}/rolling_correlations_grid.png")
    plt.close()

    # 11) Rolling correlations for key pairs
    key_pairs = [
        ('BTC-USD','ETH-USD'),      # Crypto vs Crypto
        ('BTC-USD','SPXT'),         # Crypto vs Traditional
        ('SPXT','XCMP')             # Traditional vs Traditional
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)
    for i, (c1, c2) in enumerate(key_pairs):
        rc = data[c1].rolling(window).corr(data[c2])
        axes[i].plot(rc, label=f"{c1} & {c2}", color='royalblue', linewidth=2)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[i].axvspan(peak, crash_end, color='lightgray', alpha=0.3)
        axes[i].set_title(f"{c1} vs {c2}\n(30-day rolling)")
        axes[i].set_ylim(-1,1)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(f"{out_dir}/rolling_correlations_key_pairs.png")
    plt.close()

    # 12) Save final imputed log returns for Part 2
    log_ret.to_csv(out_path)

    return log_ret

if __name__ == "__main__":
    prepare_data()