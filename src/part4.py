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

        # compute δ1‐distance matrix
        dist = schweizer_wolf_distance(window)

        # hierarchical clustering
        condensed = squareform(dist.values)
        link      = linkage(condensed, method="single")

        # get sorted index and then labels
        order  = getQuasiDiag(link)
        labels = window.columns[order].tolist()

        # compute HRP on the *covariance* of window
        cov    = window.cov()
        w      = getRecBipart(cov, labels)

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
    Rebalance monthly: signal from past 12 months, then weight = HRPe * sign(signal),
    normalized so sum(abs(weights)) = 1. Returns DataFrame of monthly weights.
    """
    lookback         = cfg["windows"]["lookback"]
    momentum_months  = cfg.get("tsm", {}).get("lookback_months", 12)
    dates            = log_returns.resample('ME').last().index
    out_dir          = os.path.dirname(cfg["paths"]["log_returns_csv"])
    os.makedirs(out_dir, exist_ok=True)

    weights_ts = []
    index_dates = []

    for date in dates:
        # need at least momentum_months of history
        if date < log_returns.index[0] + pd.DateOffset(months=momentum_months):
            continue

        # 1) time‐series momentum signal
        past   = date - pd.DateOffset(months=momentum_months)
        recent = date - pd.DateOffset(months=1)
        ret    = (log_returns.loc[recent] - log_returns.loc[past])
        signal = np.sign(ret)

        # 2) HRPe weights as above
        window = log_returns.loc[:date].tail(lookback)
        dist   = schweizer_wolf_distance(window)
        link   = linkage(squareform(dist.values), method="single")
        order  = getQuasiDiag(link)
        labels = window.columns[order].tolist()
        hrp_w  = getRecBipart(window.cov(), labels)

        # 3) apply signal, normalize
        w = hrp_w * signal.reindex(hrp_w.index)
        w = w.div(w.abs().sum())

        weights_ts.append(w)
        index_dates.append(date)

    weights_df = pd.DataFrame(weights_ts, index=index_dates)
    weights_df.to_csv(f"{out_dir}/weights_TSM_HRPe.csv")

    # heatmap over time
    plt.figure(figsize=(12,8))
    sns.heatmap(weights_df.T, cmap='RdBu_r', center=0)
    plt.title('TSM-HRPe Weights Through Time')
    plt.xlabel('Rebalance Date')
    plt.ylabel('Asset')
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