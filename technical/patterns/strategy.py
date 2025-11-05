"""Candlestick pattern recognition strategy"""

from typing import Tuple
from core.models import MarketData
from signal_evaluation.candlesticks import get_candlestick_score


class CandlestickPatternStrategy:
    """Candlestick pattern recognition strategy"""
    
    def __init__(self, weight: float = 0.10):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "Candlestick Patterns"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """Evaluate candlestick patterns from historical data"""
        # Use existing candlestick scoring logic
        score, reason = get_candlestick_score(data.historical)
        
        # Normalize score from -1...1 to 0...1
        normalized_score = (score + 1) / 2
        
        return normalized_score, reason

