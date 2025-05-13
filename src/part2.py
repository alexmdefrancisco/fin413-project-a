# Part 2: Portfolio Contruction and Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from scipy.linalg import eigh
from numpy.linalg import eigvalsh
import os

# Set plotting style
sns.set(style='whitegrid')

def compute_covariances(cfg: dict, lr: pd.DataFrame) -> tuple:
    """
    Computes sample covariance matrices for peak and trough dates using log returns,
    saves them to CSV files, and returns them.
    
    Args:
        cfg (dict): Configuration dictionary from config.yaml
        lr (pd.DataFrame): Log returns DataFrame from part1
    
    Returns:
        tuple: (cov_pp, cov_tr) - Covariance matrices at peak and trough
    """
    # Extract dates and window size from config
    datePP = pd.to_datetime(cfg["dates"]["peak"])
    dateTr = pd.to_datetime(cfg["dates"]["trough"])
    window_size = cfg["windows"]["lookback"]

    # Validate dates
    if datePP not in lr.index or dateTr not in lr.index:
        raise ValueError("Specified dates not in log returns index")

    # Extract windows
    window_pp = lr.loc[:datePP].tail(window_size)
    window_tr = lr.loc[:dateTr].tail(window_size)

    # Compute covariance matrices
    cov_pp = window_pp.cov()
    cov_tr = window_tr.cov()

    # Save to files
    out_dir = os.path.dirname(cfg["paths"]["cov_pp_csv"])
    os.makedirs(out_dir, exist_ok=True)
    cov_pp.to_csv(cfg["paths"]["cov_pp_csv"])
    cov_tr.to_csv(cfg["paths"]["cov_tr_csv"])

    return cov_pp, cov_tr

def analyze_covariances(cfg: dict) -> None:
    """
    Performs portfolio analysis including EW portfolio values, covariance analysis,
    cleaning, Euler risk contributions, diversification distribution, and rank comparison.
    Saves all figures and tables to the output directory.
    
    Args:
        cfg (dict): Configuration dictionary from config.yaml
    """
    # Load raw data for prices
    raw_data_path = cfg["data"]["raw_data"]
    data = pd.read_csv(raw_data_path, sep=',')
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    data.set_index('Date', inplace=True)
    
    # Define assets
    crypto_names = ['ADA-USD', 'BCH-USD', 'BTC-USD', 'DOGE-USD', 'ETH-USD', 
                    'LINK-USD', 'LTC-USD', 'MANA-USD', 'XLM-USD', 'XRP-USD']
    traditional_names = ['SPXT', 'XCMP', 'SOFR', 'VIX']
    all_assets = crypto_names + traditional_names
    prices = data[all_assets].ffill().dropna()

    # Output directory
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # --- Section 2.a: EW Portfolio Values ---
    datePP = pd.to_datetime(cfg["dates"]["peak"])
    dateTr = pd.to_datetime(cfg["dates"]["trough"])
    if datePP not in prices.index or dateTr not in prices.index:
        raise ValueError("Specified dates not in prices index")

    n_assets = len(all_assets)
    equal_weights = np.array([1/n_assets] * n_assets)
    prices_at_PP = prices.loc[datePP, all_assets]
    prices_at_Tr = prices.loc[dateTr, all_assets]
    portfolio_value_PP = np.dot(equal_weights, prices_at_PP)
    portfolio_value_Tr = np.dot(equal_weights, prices_at_Tr)
    
    print(f"Equally Weighted Portfolio Value on {datePP.date()}: ${portfolio_value_PP:.2f}")
    print(f"Equally Weighted Portfolio Value on {dateTr.date()}: ${portfolio_value_Tr:.2f}")

    # Pie charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(equal_weights, labels=all_assets, autopct='%1.1f%%')
    axes[0].set_title(f'Equal Weight Portfolio Composition\nas of {datePP.date()}')
    axes[1].pie(equal_weights, labels=all_assets, autopct='%1.1f%%')
    axes[1].set_title(f'Equal Weight Portfolio Composition\nas of {dateTr.date()}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ew_portfolio_composition.png")
    plt.close()

    # --- Section 2.b: Load Covariance Matrices ---
    cov_pp = pd.read_csv(cfg["paths"]["cov_pp_csv"], index_col=0)
    cov_tr = pd.read_csv(cfg["paths"]["cov_tr_csv"], index_col=0)

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cov_pp, ax=axes[0], cmap="coolwarm", annot=False, fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.7})
    axes[0].set_title("Covariance Matrix - Previous Peak")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)
    sns.heatmap(cov_tr, ax=axes[1], cmap="coolwarm", annot=False, fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.7})
    axes[1].set_title("Covariance Matrix - Trough")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/covariance_heatmaps.png")
    plt.close()

    # --- Section 2.c: Clean Covariance Matrices ---
    def clean_covariance_matrix(cov_matrix, clip_threshold=1e-4):
        eigvals, eigvecs = eigh(cov_matrix)
        eigvals[eigvals < clip_threshold] = clip_threshold
        cleaned_cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return cleaned_cov_matrix, eigvals, eigvecs

    cleaned_cov_pp, eigvals_pp, eigvecs_pp = clean_covariance_matrix(cov_pp.values)
    cleaned_cov_tr, eigvals_tr, eigvecs_tr = clean_covariance_matrix(cov_tr.values)

    # Save cleaned matrices
    np.save(cfg["paths"]["cleaned_cov_pp_npy"], cleaned_cov_pp)
    np.save(cfg["paths"]["cleaned_cov_tr_npy"], cleaned_cov_tr)

    # Eigenvalue spectra
    def plot_eigenvalue_spectra(cov_matrix, cleaned_cov_matrix, title, filename):
        eigvals_raw = eigvalsh(cov_matrix)[::-1]
        eigvals_clean = eigvalsh(cleaned_cov_matrix)[::-1]
        plt.figure(figsize=(12, 6))
        sns.kdeplot(eigvals_raw, label='Raw Eigenvalues', color='b', fill=True, alpha=0.5)
        sns.kdeplot(eigvals_clean, label='Cleaned Eigenvalues', color='r', fill=True, alpha=0.5)
        plt.title(f'Eigenvalue Spectra - {title}')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    plot_eigenvalue_spectra(cov_pp.values, cleaned_cov_pp, "Previous Peak", f"{out_dir}/eigenvalue_spectra_pp.png")
    plot_eigenvalue_spectra(cov_tr.values, cleaned_cov_tr, "Trough", f"{out_dir}/eigenvalue_spectra_tr.png")

    # 3D eigenvalue plot
    def plot_3d_eigenvalues(cov_matrix, cleaned_cov_matrix, title, filename):
        eigvals_raw = eigvalsh(cov_matrix)[::-1]
        eigvals_clean = eigvalsh(cleaned_cov_matrix)[::-1]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(range(len(eigvals_raw)), eigvals_raw, zs=0, zdir='z', label='Raw', color='b', s=50)
        ax.scatter(range(len(eigvals_clean)), eigvals_clean, zs=1, zdir='z', label='Cleaned', color='r', s=50)
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_zlabel('Value')
        ax.set_title(f'3D Eigenvalues - {title}')
        ax.legend()
        plt.savefig(filename)
        plt.close()

    plot_3d_eigenvalues(cov_pp.values, cleaned_cov_pp, "Previous Peak", f"{out_dir}/3d_eigenvalues_pp.png")
    plot_3d_eigenvalues(cov_tr.values, cleaned_cov_tr, "Trough", f"{out_dir}/3d_eigenvalues_tr.png")

    # --- Section 2.d: Euler Risk Contributions ---
    def calculate_euler_risk_contributions(weights, covariance_matrix):
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        marginal_contributions = covariance_matrix @ weights
        euler_rc = weights * marginal_contributions / portfolio_std
        return euler_rc, portfolio_std

    def herfindahl_index(risk_contributions, portfolio_std):
        normalized = risk_contributions / portfolio_std
        return np.sum(normalized**2)

    rc_pp_raw, std_pp_raw = calculate_euler_risk_contributions(equal_weights, cov_pp.values)
    h_index_pp_raw = herfindahl_index(rc_pp_raw, std_pp_raw)
    rc_tr_raw, std_tr_raw = calculate_euler_risk_contributions(equal_weights, cov_tr.values)
    h_index_tr_raw = herfindahl_index(rc_tr_raw, std_tr_raw)
    rc_pp_clean, std_pp_clean = calculate_euler_risk_contributions(equal_weights, cleaned_cov_pp)
    h_index_pp_clean = herfindahl_index(rc_pp_clean, std_pp_clean)
    rc_tr_clean, std_tr_clean = calculate_euler_risk_contributions(equal_weights, cleaned_cov_tr)
    h_index_tr_clean = herfindahl_index(rc_tr_clean, std_tr_clean)

    print("\n--- Herfindahl Indices ---")
    print(f"PP Raw: {h_index_pp_raw:.4f}")
    print(f"PP Cleaned: {h_index_pp_clean:.4f}")
    print(f"Tr Raw: {h_index_tr_raw:.4f}")
    print(f"Tr Cleaned: {h_index_tr_clean:.4f}")

    # --- Section 2.e: Diversification Distribution ---
    def diversification_distribution(rc):
        return rc / np.sum(rc)

    def entropy_enb(p):
        return np.exp(-np.sum(p * np.log(p + 1e-12)))

    def herfindahl_enb(rc):
        p = rc / np.sum(rc)
        h_index = np.sum(p**2)
        return 1 / h_index

    p_pp_raw = diversification_distribution(rc_pp_raw)
    enb_entropy_pp_raw = entropy_enb(p_pp_raw)
    enb_herfindahl_pp_raw = herfindahl_enb(rc_pp_raw)
    p_tr_raw = diversification_distribution(rc_tr_raw)
    enb_entropy_tr_raw = entropy_enb(p_tr_raw)
    enb_herfindahl_tr_raw = herfindahl_enb(rc_tr_raw)

    # Bar chart
    enb_df = pd.DataFrame({
        'Entropy ENB': [enb_entropy_pp_raw, enb_entropy_tr_raw],
        'Herfindahl ENB': [enb_herfindahl_pp_raw, enb_herfindahl_tr_raw]
    }, index=['PP Raw', 'Tr Raw'])
    enb_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Entropy vs Herfindahl ENB')
    plt.ylabel('ENB')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/enb_comparison.png")
    plt.close()

    # Normalized risk contributions
    rc_df = pd.DataFrame({
        'Asset': all_assets,
        'PP Raw RC': p_pp_raw,
        'Tr Raw RC': p_tr_raw
    })
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Asset', y='PP Raw RC', data=rc_df, color='b', label='PP Raw')
    sns.barplot(x='Asset', y='Tr Raw RC', data=rc_df, color='g', label='Tr Raw', alpha=0.6)
    plt.title('Normalized Risk Contributions')
    plt.xlabel('Assets')
    plt.ylabel('Normalized RC')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/normalized_rc.png")
    plt.close()

    # --- Section 2.f: Rank Comparison ---
    price_change = (prices.loc[dateTr, all_assets] - prices.loc[datePP, all_assets]) / prices.loc[datePP, all_assets]
    losses = -price_change
    loss_ranks = losses.rank(ascending=False)
    rc_ranks = pd.Series(rc_tr_raw, index=all_assets).rank(ascending=False)
    tau, p_value = kendalltau(loss_ranks, rc_ranks)
    rank_df = pd.DataFrame({
        'Asset': all_assets,
        'Loss Rank': loss_ranks.values,
        'RC Rank': rc_ranks.values
    }).sort_values("Loss Rank")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rank_df, x='Loss Rank', y='RC Rank', hue='Asset', s=100, palette='tab10')
    plt.title(f'Loss Rank vs. RC Rank (Tau: {tau:.2f})')
    plt.xlabel("Rank by Loss")
    plt.ylabel("Rank by RC")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rank_comparison_scatter.png")
    plt.close()

    # Bar plot
    rank_df.set_index('Asset')[['Loss Rank', 'RC Rank']].plot(kind='bar', figsize=(12, 6))
    plt.title('Loss Rank vs RC Rank')
    plt.ylabel('Rank (1 = Worst)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rank_comparison_bar.png")
    plt.close()

    # Rank difference heatmap
    rank_diff = np.abs(rank_df['Loss Rank'] - rank_df['RC Rank'])
    rank_diff_matrix = pd.DataFrame(rank_diff.values, columns=['Rank Difference'], index=rank_df['Asset'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_diff_matrix, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
    plt.title('Rank Differences')
    plt.xlabel('Rank Difference')
    plt.ylabel('Assets')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rank_difference_heatmap.png")
    plt.close()

    print(f"Kendall’s Tau: {tau:.2f} (p = {p_value:.4f})")

if __name__ == "__main__":
    # For standalone testing, load config and run
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    lr = pd.read_csv(cfg["paths"]["log_returns_csv"], index_col=0)
    lr.index = pd.to_datetime(lr.index)
    compute_covariances(cfg, lr)
    analyze_covariances(cfg)