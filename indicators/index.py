import pandas as pd

from indicators.oscillators import Oscillators
from indicators.overlays import Overlays


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare technical indicators for a given DataFrame."""
    df = df.copy()

    df["rsi"] = Oscillators.rsi(df["Close"], 14)
    m_line, m_signal, m_hist = Oscillators.macd(df["Close"])
    df["macd"] = m_line
    df["macd_signal"] = m_signal
    df["macd_hist"] = m_hist
    df["atr"] = Oscillators.atr(df, 14)
    df["adx"] = Oscillators.adx(df, 14)
    df["obv_slope"] = Oscillators.on_balance_volume_slope(df)

    df["ma50"] = Overlays.sma(df["Close"], 50)
    df["ma200"] = Overlays.sma(df["Close"], 200)
    upper, mid, lower = Overlays.bollinger_bands(df["Close"], 20, 2)
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["volume_sma20"] = Overlays.sma(df["Volume"], 20)
    df["snr"] = Overlays.winshift_snr(df[["High", "Low", "Open", "Close"]])

    return df
