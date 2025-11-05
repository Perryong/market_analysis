"""Enumerations for trading signals"""

from enum import Enum


class SignalType(Enum):
    """Trading signal types with confidence levels"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish"""
        return self in [SignalType.BUY, SignalType.STRONG_BUY]
    
    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish"""
        return self in [SignalType.SELL, SignalType.STRONG_SELL]
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal requires action"""
        return self != SignalType.HOLD


class TimeFrame(Enum):
    """Analysis timeframes"""
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    
    def __str__(self) -> str:
        return self.value

