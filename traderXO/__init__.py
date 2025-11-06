"""
Trader_XO Trading Visualization System

A modular system for advanced technical analysis, order flow, and market structure visualization.
Based on Trader_XO's trading approach.
"""

from traderXO.data_manager import DataManager
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels
from traderXO.orderflow_profile import plot_orderflow_profile
from traderXO.momentum_strategy import plot_momentum_strategy
from traderXO.range_fade import plot_range_fade
from traderXO.crypto_signal_analyzer import CryptoSignalAnalyzer, analyze_crypto_pair
from traderXO.crypto_signal_formatter import CryptoSignalFormatter
from traderXO.stock_adapter import analyze_stock_with_traderxo, get_stock_data_for_traderxo

__all__ = [
    'DataManager',
    'TechnicalIndicators',
    'KeyLevels',
    'plot_orderflow_profile',
    'plot_momentum_strategy',
    'plot_range_fade',
    'CryptoSignalAnalyzer',
    'analyze_crypto_pair',
    'CryptoSignalFormatter',
    'analyze_stock_with_traderxo',
    'get_stock_data_for_traderxo'
]

__version__ = '1.2.0'

