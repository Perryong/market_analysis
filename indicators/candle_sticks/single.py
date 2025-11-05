import pandas as pd

from indicators.candle_sticks.conditions import is_bearish, is_bullish, is_downtrend, is_uptrend
from indicators.candle_sticks.helper import (
    get_candle_body,
    get_lower_wick,
    get_price_range,
    get_upper_wick,
)


class SingleCandlePatterns:

    @staticmethod
    def is_spinning_top(df: pd.DataFrame):
        """Spinning Top: Small body, long wicks."""
        body = get_candle_body(df)
        range = get_price_range(df)
        return (body / range < 0.5) & (body / range > 0.1)

    @staticmethod
    def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Doji: Open and Close are nearly equal."""
        body = get_candle_body(df)
        range = get_price_range(df)
        return body / range <= threshold

    @staticmethod
    def is_gravestone_doji(df: pd.DataFrame, threshold: float = 0.1):
        """Gravestone Doji: Long upper shadow, small body."""
        upper_wick = get_upper_wick(df)
        lower_wick = get_lower_wick(df)
        return SingleCandlePatterns.is_doji(df, threshold) & (
            upper_wick > lower_wick * 2
        )

    @staticmethod
    def is_dragonfly_doji(df: pd.DataFrame, threshold: float = 0.1):
        """Dragonfly Doji: Long lower shadow, small body."""
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return SingleCandlePatterns.is_doji(df, threshold) & (
            lower_wick > upper_wick * 2
        )

    @staticmethod
    def is_hammer(df: pd.DataFrame) -> pd.Series:
        """Hammer: Small body, long lower shadow."""
        body = get_candle_body(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (lower_wick >= 2 * body) & (upper_wick <= body)

    @staticmethod
    def is_inverted_hammer(df: pd.DataFrame) -> pd.Series:
        """Inverted Hammer: Small body, long upper shadow."""
        body = get_candle_body(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (upper_wick >= 2 * body) & (lower_wick <= body)

    @staticmethod
    def is_shooting_star(df: pd.DataFrame) -> pd.Series:
        body = get_candle_body(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (upper_wick >= 2 * body) & (lower_wick <= body)

    @staticmethod
    def is_hanging_man(df: pd.DataFrame) -> pd.Series:
        body = get_candle_body(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (lower_wick >= 2 * body) & (upper_wick <= body)

    @staticmethod
    def is_bullish_marubozu(df: pd.DataFrame, tol: float = 0.01) -> pd.Series:
        """Bullish Marubozu: Large green body, no or very short shadows."""
        range = get_price_range(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (
            is_bullish(df) & (lower_wick / range <= tol) & (upper_wick / range <= tol)
        )

    @staticmethod
    def is_bearish_marubozu(df: pd.DataFrame, tol: float = 0.01) -> pd.Series:
        """Bearish Marubozu: Large red body, no or very short shadows."""
        range = get_price_range(df)
        lower_wick = get_lower_wick(df)
        upper_wick = get_upper_wick(df)
        return (
            is_bearish(df) & (lower_wick / range <= tol) & (upper_wick / range <= tol)
        )
