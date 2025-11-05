import numpy as np
import pandas as pd


class Oscillators:

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) using exponential smoothing."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        ma_up = up.ewm(alpha=1 / length, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / length, adjust=False).mean()

        rs = ma_up / ma_down.replace(0, np.nan)  # avoid div by zero
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val.fillna(50)  # default to neutral

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
        """MACD line, Signal line, Histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def on_balance_volume_slope(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume Slope: Who's in control (Buyers vs Sellers)."""
        close = df["Close"]
        obv = [0]
        for i in range(1, len(df)):
            if close.iat[i] > close.iat[i - 1]:
                obv.append(obv[-1] + df["Volume"].iat[i])
            elif close.iat[i] < close.iat[i - 1]:
                obv.append(obv[-1] - df["Volume"].iat[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index).diff().rolling(3).mean().fillna(0)

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Average True Range (ATR)."""
        prev_close = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / length, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        # Calculate directional movements
        plus_diff = df["High"].diff()
        minus_diff = df["Low"].diff().abs()

        plus_dm = np.where((plus_diff > minus_diff) & (plus_diff > 0), plus_diff, 0.0)
        minus_dm = np.where(
            (minus_diff > plus_diff) & (df["Low"].shift() - df["Low"] > 0),
            df["Low"].shift() - df["Low"],
            0.0,
        )

        atr = Oscillators.atr(df)
        plus_di = 100 * (
            pd.Series(plus_dm, index=df.index).rolling(window=period).sum() / atr
        )
        minus_di = 100 * (
            pd.Series(minus_dm, index=df.index).rolling(window=period).sum() / atr
        )

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100

        # ADX (smoothed DX)
        return dx.rolling(window=period).mean()
