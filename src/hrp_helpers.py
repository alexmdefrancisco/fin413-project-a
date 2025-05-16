import numpy as np
import pandas as pd

def correlDist(corr: pd.DataFrame) -> pd.DataFrame:
    """
    HRP helper: correlation-based distance
        d_ij = sqrt((1 - rho_ij) / 2)
    """
    return np.sqrt((1.0 - corr) / 2.0)


def getIVP(cov: np.ndarray) -> np.ndarray:
    """
    HRP helper: inverse-variance portfolio for a covariance matrix
    """
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov_df: pd.DataFrame, items: list) -> float:
    """
    HRP helper: variance of a cluster of assets
    """
    sub_cov = cov_df.loc[items, items]
    w0 = getIVP(sub_cov.values).reshape(-1, 1)
    # scalar cluster variance
    return float(w0.T @ sub_cov.values @ w0)


def getQuasiDiag(link: np.ndarray) -> list:
    """
    HRP helper: quasi-diagonalization to get sorted indices
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df1]).sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def getRecBipart(cov_df: pd.DataFrame, sortIx: list) -> pd.Series:
    """
    HRP helper: recursive bisection to allocate weights
    """
    w = pd.Series(1.0, index=sortIx, dtype=float)
    clusters = [sortIx]
    while clusters:
        # split each cluster in two
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c)//2), (len(c)//2, len(c)))
            if len(c) > 1
        ]
        # pairwise allocate
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i+1]
            v0 = getClusterVar(cov_df, c0)
            v1 = getClusterVar(cov_df, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= (1.0 - alpha)
    return w


def delta1_copula_distance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Schweizer-Wolf δ1 distance for each pair of columns in `log_returns`.
    δ1 ≈ 12 * E[|C_hat(u,v) - u v|], using the sample-based empirical copula.
    """
    # number of observations
    n_obs, _ = log_returns.shape

    # 1) pseudo‐observations in (0,1)
    u = log_returns.rank(axis=0, method="average") / (n_obs + 1)

    assets = log_returns.columns
    D = pd.DataFrame(0.0, index=assets, columns=assets)

    for i, ai in enumerate(assets):
        ui = u[ai].values
        for j, aj in enumerate(assets[i+1:], start=i+1):
            vj = u[aj].values
            C_hat = np.array([
                np.mean((ui <= ui[k]) & (vj <= vj[k]))
                for k in range(n_obs)
            ])
            delta1 = 12.0 * np.mean(np.abs(C_hat - (ui * vj)))
            D.at[ai, aj] = D.at[aj, ai] = delta1

    return D


def delta2_copula_distance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    δ2(X,Y) = sqrt( 90 * ∫∫ |C(u,v) - u v|^2 du dv ) 
    Approximated via:
      δ2 ≈ sqrt(90 * mean_k [(Ĉ(u_k,v_k) - u_k v_k)^2])
    """
    n_obs, _ = log_returns.shape
    u = log_returns.rank(axis=0, method="average") / (n_obs + 1)
    assets = log_returns.columns
    D = pd.DataFrame(0.0, index=assets, columns=assets)

    for i, ai in enumerate(assets):
        ui = u[ai].values
        for j, aj in enumerate(assets[i+1:], start=i+1):
            vj = u[aj].values
            C_hat = np.array([
                np.mean((ui <= ui[k]) & (vj <= vj[k]))
                for k in range(n_obs)
            ])
            diff = C_hat - (ui * vj)
            delta2 = np.sqrt(90.0 * np.mean(diff**2))
            D.at[ai, aj] = D.at[aj, ai] = delta2

    return D


def delta3_copula_distance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    δ3(X,Y) = 4 * sup_{u,v ∈ [0,1]} |C(u,v) - u v|
    Approximated via:
      δ3 ≈ 4 * max_k |Ĉ(u_k,v_k) - u_k v_k|
    """
    n_obs, _ = log_returns.shape
    u = log_returns.rank(axis=0, method="average") / (n_obs + 1)
    assets = log_returns.columns
    D = pd.DataFrame(0.0, index=assets, columns=assets)

    for i, ai in enumerate(assets):
        ui = u[ai].values
        for j, aj in enumerate(assets[i+1:], start=i+1):
            vj = u[aj].values
            C_hat = np.array([
                np.mean((ui <= ui[k]) & (vj <= vj[k]))
                for k in range(n_obs)
            ])
            diff = np.abs(C_hat - (ui * vj))
            delta3 = 4.0 * np.max(diff)
            D.at[ai, aj] = D.at[aj, ai] = delta3

    return D


def schweizer_wolf_distance(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for the δ1 copula distance metric (Schweizer-Wolf).  
    Signature matches `delta1_copula_distance`, so you can later add δ2, δ3, etc.
    """
    return delta1_copula_distance(log_returns)