# Part 3: Risk-based Portfolios
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Utils import
from utils import load_cov

sns.set(style="whitegrid")

def build_risk_portfolios(cfg: dict, log_returns: pd.DataFrame) -> None:
    """
    Part 3: Load cleaned covariance matrices, compute:
      - Minimum-Variance portfolios
      - Equal Risk Contribution portfolios
      - Minimum ENB portfolios
    for both peak and trough dates, then save CSVs and bar-chart images.
    """
    # Unpack config
    clean_pp_path = cfg["paths"]["cleaned_cov_pp"]
    clean_tr_path = cfg["paths"]["cleaned_cov_tr"]
    out_dir = os.path.dirname(clean_pp_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load cleaned covariance matrices
    cov_pp = load_cov(clean_pp_path)
    cov_tr = load_cov(clean_tr_path)

    # Asset list from log_returns
    assets = list(log_returns.columns)
    n = len(assets)

    # Common constraints and initial guess
    w0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    cons   = ({ "type":"eq", "fun": lambda w: np.sum(w) - 1 })

    # --- 1) Minimum-Variance ---
    def portfolio_variance(w, cov):
        return w @ cov @ w

    def solve_min_variance(cov):
        res = minimize(portfolio_variance, w0, args=(cov,),
                       method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            raise ValueError("MinVar did not converge")
        return res.x

    mv_pp = solve_min_variance(cov_pp)
    mv_tr = solve_min_variance(cov_tr)

    # Save CSVs
    pd.Series(mv_pp, index=assets).to_csv(f"{out_dir}/weights_MV_peak.csv")
    pd.Series(mv_tr, index=assets).to_csv(f"{out_dir}/weights_MV_trough.csv")

    # Bar plots
    for tag, w in [("peak", mv_pp), ("trough", mv_tr)]:
        plt.figure(figsize=(12,5))
        sns.barplot(x=assets, y=w)
        plt.xticks(rotation=90)
        plt.title(f"Minimum-Variance Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_MV_{tag}.png")
        plt.close()

    # --- 2) Equal‐Risk‐Contribution ---
    def erc_objective(w, cov):
        # sum_i sum_j (RC_i - RC_j)^2
        # σ2 = w @ cov @ w
        m = cov @ w
        rc = w * m
        return np.sum((rc[:,None] - rc[None,:])**2)

    def solve_erc(cov):
        res = minimize(erc_objective, w0, args=(cov,),
                       method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            raise ValueError("ERC did not converge")
        return res.x

    erc_pp = solve_erc(cov_pp)
    erc_tr = solve_erc(cov_tr)

    pd.Series(erc_pp, index=assets).to_csv(f"{out_dir}/weights_ERC_peak.csv")
    pd.Series(erc_tr, index=assets).to_csv(f"{out_dir}/weights_ERC_trough.csv")

    for tag, w in [("peak", erc_pp), ("trough", erc_tr)]:
        plt.figure(figsize=(12,5))
        sns.barplot(x=assets, y=w)
        plt.xticks(rotation=90)
        plt.title(f"ERC Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_ERC_{tag}.png")
        plt.close()

    # --- 3) Minimum Effective Number of Bets (ENB) ---
    def enb_objective(w, cov):
        # returns negative entropy of diversification distribution
        eigvals, eigvecs = np.linalg.eigh(cov)
        # exposure to principal components
        exposures = eigvecs.T @ w
        contrib   = (exposures**2) * eigvals
        p = contrib / contrib.sum()
        # entropy
        return np.sum(p * np.log(p + 1e-12))  # minimize negative entropy

    def solve_enb(cov):
        res = minimize(enb_objective, w0, args=(cov,),
                       method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            raise ValueError("ENB optimization did not converge")
        return res.x

    enb_pp = solve_enb(cov_pp)
    enb_tr = solve_enb(cov_tr)

    pd.Series(enb_pp, index=assets).to_csv(f"{out_dir}/weights_ENB_peak.csv")
    pd.Series(enb_tr, index=assets).to_csv(f"{out_dir}/weights_ENB_trough.csv")

    for tag, w in [("peak", enb_pp), ("trough", enb_tr)]:
        plt.figure(figsize=(12,5))
        sns.barplot(x=assets, y=w)
        plt.xticks(rotation=90)
        plt.title(f"ENB-Min Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_ENB_{tag}.png")
        plt.close()