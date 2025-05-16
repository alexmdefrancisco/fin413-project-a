import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .hrp_helpers import getQuasiDiag, getRecBipart, schweizer_wolf_distance

sns.set(style="whitegrid")

def build_hrpe(cfg: dict, log_returns: pd.DataFrame) -> pd.Series:
    """
    Part 4a-b: Compute Enhanced HRP (HRPe) using true δ1 Schweizer-Wolf distance
    at peak and trough dates, save weights and bar charts.
    Returns the last weights (at trough).
    """
    lookback = cfg["windows"]["lookback"]
    peak   = pd.to_datetime(cfg["dates"]["peak"])
    trough = pd.to_datetime(cfg["dates"]["trough"])
    out_dir = os.path.dirname(cfg["paths"]["log_returns_csv"])
    os.makedirs(out_dir, exist_ok=True)

    for tag, date in [("peak", peak), ("trough", trough)]:
        # select lookback of log returns
        window = log_returns.loc[:date].tail(lookback)

        # form & clean the rolling covariance
        cov_raw = window.cov()
        eigvals, eigvecs = np.linalg.eigh(cov_raw.values)
        eigvals = np.clip(eigvals, 1e-4, None)
        cov_clean = pd.DataFrame(
            eigvecs @ np.diag(eigvals) @ eigvecs.T,
            index=cov_raw.index,
            columns=cov_raw.columns
        )

        # compute δ1-distance matrix
        dist = schweizer_wolf_distance(window)

        # cluster & quasi-diagonalize
        condensed = squareform(dist.values)
        link      = linkage(condensed, method="single")
        order     = getQuasiDiag(link)
        labels    = window.columns[order].tolist()

        # compute HRP on the *clean covariance* of window
        w = getRecBipart(cov_clean, labels)

        # save CSV and bar chart
        pd.Series(w, name="weight").to_csv(f"{out_dir}/weights_HRPe_{tag}.csv")
        plt.figure(figsize=(12,6))
        sns.barplot(x=w.index, y=w.values)
        plt.xticks(rotation=90)
        plt.title(f"HRPe Weights @ {tag.capitalize()}")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bar_HRPe_{tag}.png")
        plt.close()

    return w


def build_tsm_hrpe(cfg: dict, log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Part 4c: Integrate Time-Series Momentum (TSM) with HRPe.
    Rebalance monthly: signal from past `lookback_months`, then
    w = HRPe_weights * sign(momentum), normalized so sum(abs(w))=1.
    """
    lookback        = cfg["windows"]["lookback"]
    mom_months      = cfg.get("tsm", {}).get("lookback_months", 12)
    dates           = log_returns.resample("ME").last().index
    out_dir         = os.path.dirname(cfg["paths"]["log_returns_csv"])
    os.makedirs(out_dir, exist_ok=True)

    weights_list, idx = [], []

    for date in dates:
        # need enough history for momentum
        if date < log_returns.index[0] + pd.DateOffset(months=mom_months):
            continue

        # 1) momentum signal
        past   = date - pd.DateOffset(months=mom_months)
        recent = date - pd.DateOffset(months=1)
        ret    = (log_returns.loc[recent] - log_returns.loc[past])
        signal = np.sign(ret)

        # 2) prepare lookback window for HRPe
        window = log_returns.loc[:date].tail(lookback)

        # 2a) clean its covariance
        cov_raw = window.cov()
        eigv, eigvec = np.linalg.eigh(cov_raw.values)
        eigv = np.clip(eigv, 1e-4, None)
        cov_clean = pd.DataFrame(
            eigvec @ np.diag(eigv) @ eigvec.T,
            index=cov_raw.index, columns=cov_raw.columns
        )

        # 2b) clustering via SW on returns
        dist      = schweizer_wolf_distance(window)
        condensed = squareform(dist.values)
        link      = linkage(condensed, method="single")
        order     = getQuasiDiag(link)
        labels    = window.columns[order].tolist()

        # 2c) allocate on cleaned covariance
        hrp_w = getRecBipart(cov_clean, labels)

        # 3) tilt by momentum, normalize abs sum to 1
        w = hrp_w * signal.reindex(hrp_w.index)
        w = w.div(w.abs().sum())

        weights_list.append(w)
        idx.append(date)

    # assemble, save, plot
    weights_df = pd.DataFrame(weights_list, index=idx)
    weights_df.to_csv(f"{out_dir}/weights_TSM_HRPe.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(weights_df.T, cmap="RdBu_r", center=0)
    plt.title("TSM-HRPe Weights Over Time")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_TSM_HRPe.png")
    plt.close()

    return weights_df


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    lr  = pd.read_csv(cfg["paths"]["log_returns_csv"], index_col=0, parse_dates=True)
    build_hrpe(cfg, lr)
    build_tsm_hrpe(cfg, lr)