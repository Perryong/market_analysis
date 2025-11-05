"""Volatility-based indicator strategies (Bollinger Bands)"""

from typing import Tuple
from core.models import MarketData


class BollingerBandStrategy:
    """Bollinger Bands volatility and overbought/oversold strategy"""
    
    def __init__(self, weight: float = 0.10):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "Bollinger Bands"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate Bollinger Bands indicator"""
        close = float(data.current['Close'])
        bb_upper = float(data.current['bb_upper'])
        bb_mid = float(data.current['bb_mid'])
        bb_lower = float(data.current['bb_lower'])
        
        # Previous values for trend
        close_prev = float(data.previous['Close'])
        bb_lower_prev = float(data.previous['bb_lower'])
        bb_upper_prev = float(data.previous['bb_upper'])
        
        # Calculate band width (volatility measure)
        band_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        
        # Determine position within bands
        if close < bb_lower:
            # Below lower band - oversold
            was_below = close_prev < bb_lower_prev
            score = 0.9 if not was_below else 1.0  # Bounce expected
            label = "Strong Bullish (<lower band)"
            position = "<lower"
            
        elif close > bb_upper:
            # Above upper band - overbought
            was_above = close_prev > bb_upper_prev
            score = 0.1 if not was_above else 0.0  # Pullback expected
            label = "Strong Bearish (>upper band)"
            position = ">upper"
            
        elif close < bb_mid:
            # Between lower and middle
            distance_to_lower = (close - bb_lower) / (bb_mid - bb_lower)
            
            if distance_to_lower < 0.3:
                score, label = 0.75, "Bullish (near lower)"
            else:
                score, label = 0.6, "Mild Bullish (lower half)"
            position = "lower<->middle"
            
        elif close > bb_mid:
            # Between middle and upper
            distance_to_upper = (bb_upper - close) / (bb_upper - bb_mid)
            
            if distance_to_upper < 0.3:
                score, label = 0.25, "Bearish (near upper)"
            else:
                score, label = 0.4, "Mild Bearish (upper half)"
            position = "middle<->upper"
            
        else:
            # At middle band
            score, label = 0.5, "Neutral (at middle)"
            position = "middle"
        
        # Volatility note
        vol_note = "tight" if band_width < 0.03 else "wide" if band_width > 0.08 else "normal"
        
        explanation = (
            f"BB {label} - price {position} "
            f"({vol_note} bands, width: {band_width:.3f})"
        )
        
        return score, explanation

