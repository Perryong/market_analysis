import numpy as np
import pandas as pd


def get_price_range(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]


def get_candle_body(df: pd.DataFrame) -> pd.Series:
    return abs(df["Close"] - df["Open"])


def get_mid_body(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] + df["Close"]) / 2


def get_lower_wick(df: pd.DataFrame) -> pd.Series:
    return np.minimum(df["Open"], df["Close"]) - df["Low"]


def get_upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["High"] - np.maximum(df["Open"], df["Close"])
