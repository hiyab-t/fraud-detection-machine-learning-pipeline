from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUMERIC_SCALE = ["Time", "Amount"]

def split_x_y(df: pd.DataFrame):
    y = df["Class"].astype(int)
    x = df.drop(columns=["Class"])
    return x, y

def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = list(x.columns)
    cols_to_scale = [c for c in NUMERIC_SCALE if c in numeric]
    cols_pass = [c for c in numeric if c not in cols_to_scale]
    transformers = []
    if cols_to_scale:
        transformers.append(("scale_amt_time", StandardScaler(), cols_to_scale))
    if cols_pass:
        transformers.append(("pass", "passthrough", cols_pass))
    return ColumnTransformer(transformers=transformers)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # dataset typically has no nulls; keep it robust anyway
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    return df
