"""Volume-based indicator strategies"""

from typing import Tuple
from core.models import MarketData


class VolumeStrategy:
    """Volume confirmation strategy using SMA20"""
    
    def __init__(self, weight: float = 0.10):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "Volume Confirmation"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate volume relative to 20-day average"""
        volume = float(data.current['Volume'])
        volume_sma20 = float(data.current['volume_sma20'])
        close = float(data.current['Close'])
        close_prev = float(data.previous['Close'])
        
        # Calculate volume ratio
        if volume_sma20 > 0:
            volume_ratio = volume / volume_sma20
        else:
            volume_ratio = 1.0
        
        # Determine price direction
        price_up = close > close_prev
        price_down = close < close_prev
        
        # Scoring based on volume confirmation
        if volume_ratio > 2.0:
            # Very high volume
            if price_up:
                score, label = 1.0, "Strong Bullish (high volume surge)"
            elif price_down:
                score, label = 0.0, "Strong Bearish (high volume dump)"
            else:
                score, label = 0.5, "Neutral (high volume, no direction)"
        
        elif volume_ratio > 1.3:
            # Above average volume
            if price_up:
                score, label = 0.8, "Bullish (above avg volume)"
            elif price_down:
                score, label = 0.2, "Bearish (above avg volume)"
            else:
                score, label = 0.5, "Neutral (elevated volume)"
        
        elif volume_ratio > 0.7:
            # Normal volume
            if price_up:
                score, label = 0.6, "Mild Bullish (normal volume)"
            elif price_down:
                score, label = 0.4, "Mild Bearish (normal volume)"
            else:
                score, label = 0.5, "Neutral (normal volume)"
        
        else:
            # Below average volume (weak confirmation)
            if price_up:
                score, label = 0.55, "Weak Bullish (low volume)"
            elif price_down:
                score, label = 0.45, "Weak Bearish (low volume)"
            else:
                score, label = 0.5, "Neutral (low volume)"
        
        explanation = (
            f"Volume {volume:,.0f} vs SMA20 {volume_sma20:,.0f} - "
            f"{label} (ratio: {volume_ratio:.2f}x)"
        )
        
        return score, explanation

