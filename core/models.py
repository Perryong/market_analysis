"""Data models for trading signal system"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

from core.enums import SignalType, TimeFrame


@dataclass
class MarketData:
    """Encapsulated market data for analysis"""
    ticker: str
    current: pd.Series
    previous: pd.Series
    historical: pd.DataFrame
    timeframe: TimeFrame
    short_name: Optional[str] = None
    
    @property
    def last_close(self) -> float:
        """Get most recent closing price"""
        return float(self.current['Close'])
    
    @property
    def atr(self) -> float:
        """Get Average True Range"""
        return float(self.current['atr'])
    
    @property
    def adx(self) -> float:
        """Get Average Directional Index"""
        return float(self.current['adx'])
    
    @property
    def rsi(self) -> float:
        """Get Relative Strength Index"""
        return float(self.current['rsi'])
    
    @property
    def snr(self) -> float:
        """Get Support/Resistance level"""
        value = self.current['snr']
        return float(value) if not np.isnan(value) else np.nan


@dataclass
class EntryZone:
    """Suggested entry price range based on ATR"""
    lower_bound: float
    upper_bound: float
    reference_price: float
    volatility: float  # ATR value
    
    def __str__(self) -> str:
        return f"${self.lower_bound:.2f} - ${self.upper_bound:.2f}"
    
    @property
    def range_percent(self) -> float:
        """Calculate range as percentage of reference price"""
        range_size = self.upper_bound - self.lower_bound
        return (range_size / self.reference_price) * 100
    
    @classmethod
    def from_atr(cls, close_price: float, atr: float, 
                 lower_mult: float = 0.5, upper_mult: float = 0.8) -> 'EntryZone':
        """Create entry zone from ATR calculation"""
        return cls(
            lower_bound=close_price - (lower_mult * atr),
            upper_bound=close_price + (upper_mult * atr),
            reference_price=close_price,
            volatility=atr
        )


@dataclass
class TradingSignal:
    """Complete trading signal with analysis results"""
    ticker: str
    short_name: str
    signal: SignalType
    confidence: float  # 0.0 to 1.0
    reasons: List[str] = field(default_factory=list)
    entry_zone: Optional[EntryZone] = None
    last_close: float = 0.0
    atr: float = 0.0
    timeframe: TimeFrame = TimeFrame.DAILY
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on"""
        return self.signal.is_actionable
    
    @property
    def confidence_percent(self) -> float:
        """Get confidence as percentage"""
        return self.confidence * 100
    
    def __str__(self) -> str:
        """String representation of signal"""
        return f"{self.ticker}: {self.signal} ({self.confidence_percent:.1f}%)"

