"""Base protocol for indicator strategies"""

from typing import Protocol, Tuple
from core.models import MarketData


class IndicatorStrategy(Protocol):
    """Protocol defining interface for all indicator strategies"""
    
    @property
    def name(self) -> str:
        """Strategy name for identification"""
        ...
    
    @property
    def weight(self) -> float:
        """Weight of this strategy in final score (0.0 to 1.0)"""
        ...
    
    def evaluate(self, data: MarketData) -> Tuple[float, str]:
        """
        Evaluate indicator and return (score, explanation)
        
        Args:
            data: Market data with current, previous, and historical prices
            
        Returns:
            Tuple of (score, explanation) where:
            - score: 0.0 (bearish) to 1.0 (bullish)
            - explanation: Human-readable reason for the score
        """
        ...

