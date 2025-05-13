import numpy as np
import pandas as pd

def load_cov(path: str) -> np.ndarray:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, index_col=0).values
    return np.load(path)