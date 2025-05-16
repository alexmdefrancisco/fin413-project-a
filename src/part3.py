import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from utils import load_cov
from .hrp_helpers import correlDist, getQuasiDiag, getRecBipart

sns.set(style="whitegrid")

def build_risk_portfolios(cfg: dict, log_returns: pd.DataFrame) -> None:
    """
    Part 3: Load cleaned covariance matrices and compute:
      1) Minimum-Variance
      2) Equal-Risk-Contribution
      3) Minimum ENB
      4) Hierarchical Risk Parity (HRP)
    at both peak and trough dates, saving weights CSVs and bar plots.
    """
    # — load cleaned covariances —
    cov_pp = load_cov(cfg["paths"]["cleaned_cov_pp"])
    cov_tr = load_cov(cfg["paths"]["cleaned_cov_tr"])

    # assets & DataFrame wrappers
    assets    = list(log_returns.columns)
    cov_pp_df = pd.DataFrame(cov_pp, index=assets, columns=assets)
    cov_tr_df = pd.DataFrame(cov_tr, index=assets, columns=assets)

    out_dir = os.path.dirname(cfg["paths"]["cleaned_cov_pp"])
    os.makedirs(out_dir, exist_ok=True)

    n      = len(assets)
    w0     = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    cons   = ({ "type":"eq", "fun": lambda w: w.sum() - 1 })

    # 1) Minimum-Variance
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
    pd.Series(mv_pp, index=assets).to_csv(f"{out_dir}/weights_MV_peak.csv")
    pd.Series(mv_tr, index=assets).to_csv(f"{out_dir}/weights_MV_trough.csv")

    for tag, w in [("peak", mv_pp), ("trough", mv_tr)]:
        plt.figure(figsize=(12,5))
        sns.barplot(x=assets, y=w)
        plt.xticks(rotation=90)
        plt.title(f"Minimum-Variance Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_MV_{tag}.png")
        plt.close()

    # 2) Equal-Risk-Contribution
    def erc_objective(w, cov):
        m  = cov @ w
        rc = w * m
        return ((rc[:,None] - rc[None,:])**2).sum()

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

    # 3) Minimum ENB
    def enb_objective(w, cov):
        eigvals, eigvecs = np.linalg.eigh(cov)
        exposures = eigvecs.T @ w
        contrib   = exposures**2 * eigvals
        p         = contrib / contrib.sum()
        return (p * np.log(p + 1e-12)).sum()

    def solve_enb(cov):
        res = minimize(enb_objective, w0, args=(cov,),
                       method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            raise ValueError("ENB did not converge")
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

    # 4) Hierarchical Risk Parity (HRP)
    def solve_hrp(cov_df, clean_cov_matrix=True):
        cov0 = cov_df.copy()
        if clean_cov_matrix:
            eigvals, eigvecs = np.linalg.eigh(cov0.values)
            eigvals = np.clip(eigvals, 1e-4, None)
            cov0    = pd.DataFrame(
                eigvecs @ np.diag(eigvals) @ eigvecs.T,
                index=cov0.index, columns=cov0.columns
            )
        corr = cov0.corr()
        dist = correlDist(corr)
        link = sch.linkage(squareform(dist.values), method="single")
        sortIx = getQuasiDiag(link)
        ordered = corr.index[sortIx].tolist()
        return getRecBipart(cov0, ordered)

    hrp_pp = solve_hrp(cov_pp_df)
    hrp_tr = solve_hrp(cov_tr_df)
    pd.Series(hrp_pp, index=assets).to_csv(f"{out_dir}/weights_HRP_peak.csv")
    pd.Series(hrp_tr, index=assets).to_csv(f"{out_dir}/weights_HRP_trough.csv")

    for tag, w in [("peak", hrp_pp), ("trough", hrp_tr)]:
        plt.figure(figsize=(12,5))
        sns.barplot(x=assets, y=w.values)
        plt.xticks(rotation=90)
        plt.title(f"HRP Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_HRP_{tag}.png")
        plt.close()


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    lr  = pd.read_csv(cfg["paths"]["log_returns_csv"], index_col=0, parse_dates=True)
    build_risk_portfolios(cfg, lr)