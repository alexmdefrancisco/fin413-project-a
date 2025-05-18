import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        # clean covariance
        cov_raw = window.cov()
        eigvals, eigvecs = np.linalg.eigh(cov_raw.values)
        eigvals = np.clip(eigvals, 1e-4, None)
        cov_clean = pd.DataFrame(
            eigvecs @ np.diag(eigvals) @ eigvecs.T,
            index=cov_raw.index,
            columns=cov_raw.columns
        )
        # build distance on returns
        dist = schweizer_wolf_distance(window)

        # cluster & quasi-diagonalize
        condensed = squareform(dist.values)
        link      = linkage(condensed, method="single")
        order     = getQuasiDiag(link)
        labels    = window.columns[order].tolist()
        # allocate HRP on cleaned covariance
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
    Returns monthly weights DataFrame.
    """
    lookback        = cfg["windows"]["lookback"]
    mom_months      = cfg.get("tsm", {}).get("lookback_months", 12)
    dates           = log_returns.resample("ME").last().index
    out_dir         = os.path.dirname(cfg["paths"]["log_returns_csv"])
    os.makedirs(out_dir, exist_ok=True)

    weights_list, idx = [], []
    for date in dates:
        if date < log_returns.index[0] + pd.DateOffset(months=mom_months):
            continue

        # 1) momentum signal
        past   = date - pd.DateOffset(months=mom_months)
        recent = date - pd.DateOffset(months=1)
        # cumulative log-return signal
        signal = np.sign(log_returns.loc[recent] - log_returns.loc[past])
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
        # 2b) cluster via SW distance
        dist      = schweizer_wolf_distance(window)
        condensed = squareform(dist.values)
        link      = linkage(condensed, method="single")
        order     = getQuasiDiag(link)
        labels    = window.columns[order].tolist()

        # 2c) allocate HRP on cleaned covariance
        hrp_w     = getRecBipart(cov_clean, labels)
        # 3) tilt by momentum
        w = hrp_w * signal.reindex(hrp_w.index)
        w = w.div(w.abs().sum())
        weights_list.append(w)
        idx.append(date)

    weights_df = pd.DataFrame(weights_list, index=idx)
    weights_df.to_csv(f"{out_dir}/weights_TSM_HRPe.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(weights_df.T, cmap="RdBu_r", center=0)
    step = 2  # Show every 2nd date
    formatted_labels = [date.strftime("%Y-%m-%d") for date in weights_df.index[::step]]
    plt.xticks(ticks=range(0, len(weights_df.index), step), labels=formatted_labels, rotation=90)    
    plt.title("TSM-HRPe Weights Over Time")
    plt.xlabel("Rebalance Month")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_TSM_HRPe.png")
    plt.close()

    return weights_df


def compute_performance(cfg: dict, log_returns: pd.DataFrame) -> None:
    """
    Compute and plot cumulative wealth and summary metrics
    for 1/N, HRPe, and TSM-HRPe strategies.
    """
    out_dir = os.path.dirname(cfg["paths"]["log_returns_csv"])
    os.makedirs(out_dir, exist_ok=True)

    # monthly log-returns
    ret_m = log_returns.resample("ME").sum()

    assets = log_returns.columns
    dates  = ret_m.index

    # 1/N weights
    w_eq = pd.DataFrame(1/len(assets), index=dates, columns=assets)

    # HRPe static at trough
    w_hrpe_single = pd.read_csv(f"{out_dir}/weights_HRPe_trough.csv", index_col=0).squeeze()
    w_hrpe = pd.DataFrame(np.tile(w_hrpe_single.values, (len(dates),1)),
                          index=dates, columns=assets)

    # TSM-HRPe monthly
    w_tsm = pd.read_csv(f"{out_dir}/weights_TSM_HRPe.csv", index_col=0, parse_dates=True)

    strategies = {
        "1/N":    w_eq,
        "HRPe":   w_hrpe,
        "TSM-HRPe": w_tsm
    }

    perf = {}
    for name, w in strategies.items():
        w_aligned, r_aligned = w.align(ret_m, join="inner", axis=0)
        p_ret = (w_aligned * r_aligned).sum(axis=1)
        wealth = np.exp(p_ret.cumsum())
        ann_ret = p_ret.mean() * 12
        ann_vol = p_ret.std()  * np.sqrt(12)
        sharpe  = ann_ret / ann_vol
        perf[name] = {"wealth": wealth,
                      "ann_ret": ann_ret,
                      "ann_vol": ann_vol,
                      "sharpe": sharpe}

    # plot equity curves
    plt.figure(figsize=(10,6))
    for name, v in perf.items():
        plt.plot(v["wealth"], label=name)
    plt.legend()
    plt.title("Cumulative Wealth")
    plt.xlabel("Date")
    plt.ylabel("Wealth (base = 1)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/wealth_comparison.png")
    plt.close()

    # summary table
    summary = pd.DataFrame({
        name: {"Ann. Return": v["ann_ret"],
               "Ann. Vol":    v["ann_vol"],
               "Sharpe":      v["sharpe"]}
        for name, v in perf.items()
    }).T
    summary.to_csv(f"{out_dir}/summary_performance.csv")

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    lr  = pd.read_csv(cfg["paths"]["log_returns_csv"], index_col=0, parse_dates=True)
    build_hrpe(cfg, lr)
    build_tsm_hrpe(cfg, lr)
    compute_performance(cfg, lr)