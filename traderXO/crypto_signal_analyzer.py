"""
Cryptocurrency Signal Analyzer
Generates BUY/SELL/HOLD signals with confidence scores for crypto assets
"""

import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

from traderXO.data_manager import DataManager
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels


@dataclass
class CryptoSignal:
    """Crypto trading signal with analysis"""
    ticker: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 100.0
    last_close: float
    atr: float
    entry_zone_low: float
    entry_zone_high: float
    analysis_details: List[str]
    market_structure: Dict[str, float]


class CryptoSignalAnalyzer:
    """Analyze cryptocurrency and generate trading signals"""
    
    def __init__(self):
        self.weights = {
            'rsi': 0.20,
            'macd': 0.15,
            'delta': 0.15,
            'ema_trend': 0.20,
            'volume': 0.10,
            'market_profile': 0.10,
            'momentum': 0.10
        }
    
    def analyze(self, ticker: str, df: pd.DataFrame, weekly_df: pd.DataFrame) -> CryptoSignal:
        """
        Analyze cryptocurrency and generate signal
        
        Args:
            ticker: Trading pair symbol
            df: Main timeframe data with indicators
            weekly_df: Weekly timeframe data
            
        Returns:
            CryptoSignal with analysis
        """
        # Get latest data
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Calculate scores
        rsi_score, rsi_analysis = self._analyze_rsi(current)
        macd_score, macd_analysis = self._analyze_macd(current)
        delta_score, delta_analysis = self._analyze_delta(df.tail(20))
        ema_score, ema_analysis = self._analyze_ema_trend(current)
        volume_score, volume_analysis = self._analyze_volume(current, df)
        profile_score, profile_analysis = self._analyze_market_profile(current)
        momentum_score, momentum_analysis = self._analyze_momentum(df.tail(10))
        
        # Calculate weighted score
        total_score = (
            rsi_score * self.weights['rsi'] +
            macd_score * self.weights['macd'] +
            delta_score * self.weights['delta'] +
            ema_score * self.weights['ema_trend'] +
            volume_score * self.weights['volume'] +
            profile_score * self.weights['market_profile'] +
            momentum_score * self.weights['momentum']
        )
        
        # Determine signal
        if total_score > 0.6:
            signal = 'BUY'
        elif total_score < -0.6:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Confidence (0-100)
        confidence = abs(total_score) * 100
        
        # Entry zone (based on ATR)
        atr = current['atr']
        entry_low = current['close'] - (atr * 0.5)
        entry_high = current['close'] + (atr * 0.8)
        
        # Compile analysis
        analysis_details = [
            rsi_analysis,
            macd_analysis,
            delta_analysis,
            ema_analysis,
            volume_analysis,
            profile_analysis,
            momentum_analysis
        ]
        
        # Market structure
        market_structure = {
            'rsi': current['rsi'],
            'macd': current['macd'],
            'delta': current['delta'],
            'cumulative_delta': current['cumulative_delta'],
            'ema_12': current.get('ema_12_weekly', 0),
            'ema_21': current.get('ema_21_weekly', 0),
            'volume': current['volume'],
            'atr': atr
        }
        
        return CryptoSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            last_close=current['close'],
            atr=atr,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            analysis_details=analysis_details,
            market_structure=market_structure
        )
    
    def _analyze_rsi(self, current: pd.Series) -> Tuple[float, str]:
        """Analyze RSI indicator"""
        rsi = current['rsi']
        
        if rsi < 30:
            score = 1.0
            signal = "Strong Bullish (oversold)"
        elif rsi < 40:
            score = 0.5
            signal = "Bullish"
        elif rsi > 70:
            score = -1.0
            signal = "Strong Bearish (overbought)"
        elif rsi > 60:
            score = -0.5
            signal = "Bearish"
        else:
            score = 0.0
            signal = "Neutral"
        
        analysis = f"RSI {rsi:.1f} - {signal}"
        return score, analysis
    
    def _analyze_macd(self, current: pd.Series) -> Tuple[float, str]:
        """Analyze MACD indicator"""
        macd = current['macd']
        macd_signal = current['macd_signal']
        macd_hist = current['macd_hist']
        
        # MACD crossover
        if macd > macd_signal and macd_hist > 0:
            score = 0.8
            signal = "Bullish Crossover"
        elif macd < macd_signal and macd_hist < 0:
            score = -0.8
            signal = "Bearish Crossover"
        elif macd > 0:
            score = 0.3
            signal = "Bullish Momentum"
        elif macd < 0:
            score = -0.3
            signal = "Bearish Momentum"
        else:
            score = 0.0
            signal = "Neutral"
        
        spread = macd - macd_signal
        analysis = f"MACD {signal} (spread: {spread:.4f}, hist: {macd_hist:.4f})"
        return score, analysis
    
    def _analyze_delta(self, recent_df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze order flow delta"""
        avg_delta = recent_df['delta'].mean()
        cumulative_delta = recent_df['cumulative_delta'].iloc[-1]
        delta_trend = recent_df['cumulative_delta'].diff().tail(5).mean()
        
        if avg_delta > 0 and delta_trend > 0:
            score = 0.8
            signal = "Strong Buying Pressure"
        elif avg_delta > 0:
            score = 0.4
            signal = "Buying Pressure"
        elif avg_delta < 0 and delta_trend < 0:
            score = -0.8
            signal = "Strong Selling Pressure"
        elif avg_delta < 0:
            score = -0.4
            signal = "Selling Pressure"
        else:
            score = 0.0
            signal = "Balanced"
        
        analysis = f"Delta {signal} (avg: {avg_delta:.0f}, cumulative: {cumulative_delta:.0f})"
        return score, analysis
    
    def _analyze_ema_trend(self, current: pd.Series) -> Tuple[float, str]:
        """Analyze EMA trend"""
        if 'ema_12_weekly' not in current or 'ema_21_weekly' not in current:
            return 0.0, "EMA Trend - Not Available"
        
        price = current['close']
        ema_12 = current['ema_12_weekly']
        ema_21 = current['ema_21_weekly']
        ema_20 = current['ema_20']
        ema_50 = current['ema_50']
        
        # Weekly EMAs (primary trend)
        if price > ema_12 > ema_21:
            score = 1.0
            signal = "Strong Bullish"
        elif price > ema_12:
            score = 0.6
            signal = "Bullish"
        elif price < ema_21 < ema_12:
            score = -1.0
            signal = "Strong Bearish"
        elif price < ema_21:
            score = -0.6
            signal = "Bearish"
        else:
            score = 0.0
            signal = "Mixed"
        
        # Intraday EMAs (secondary confirmation)
        if ema_20 > ema_50:
            score += 0.2
        elif ema_20 < ema_50:
            score -= 0.2
        
        spread = ((ema_12 - ema_21) / ema_21 * 100) if ema_21 > 0 else 0
        analysis = f"EMA Trend {signal} (12: {ema_12:.2f}, 21: {ema_21:.2f}, spread: {spread:.1f}%)"
        return score, analysis
    
    def _analyze_volume(self, current: pd.Series, df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze volume"""
        volume = current['volume']
        volume_avg = current['volume_avg']
        volume_ratio = volume / volume_avg if volume_avg > 0 else 1
        
        # Volume + price direction
        price_change = current['close'] - current['open']
        
        if volume_ratio > 2.0:
            if price_change > 0:
                score = 0.8
                signal = "Strong Bullish (high volume up)"
            else:
                score = -0.8
                signal = "Strong Bearish (high volume down)"
        elif volume_ratio > 1.5:
            if price_change > 0:
                score = 0.4
                signal = "Bullish (above avg volume)"
            else:
                score = -0.4
                signal = "Bearish (above avg volume)"
        else:
            score = 0.0
            signal = "Weak (low volume)"
        
        analysis = f"Volume {signal} (ratio: {volume_ratio:.2f}x avg)"
        return score, analysis
    
    def _analyze_market_profile(self, current: pd.Series) -> Tuple[float, str]:
        """Analyze market profile levels"""
        if 'poc' not in current or 'vah' not in current or 'val' not in current:
            return 0.0, "Market Profile - Not Available"
        
        price = current['close']
        poc = current['poc']
        vah = current['vah']
        val = current['val']
        
        # Position relative to value area
        if price > vah:
            score = 0.5
            signal = "Above Value Area (bullish)"
        elif price < val:
            score = -0.5
            signal = "Below Value Area (bearish)"
        elif price > poc:
            score = 0.2
            signal = "Above POC (mild bullish)"
        elif price < poc:
            score = -0.2
            signal = "Below POC (mild bearish)"
        else:
            score = 0.0
            signal = "At POC (balanced)"
        
        analysis = f"Market Profile {signal} (POC: {poc:.2f})"
        return score, analysis
    
    def _analyze_momentum(self, recent_df: pd.DataFrame) -> Tuple[float, str]:
        """Analyze price momentum"""
        # Calculate rate of change
        roc = ((recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / 
               recent_df['close'].iloc[0] * 100)
        
        # ATR expansion/compression
        atr_current = recent_df['atr'].iloc[-1]
        atr_avg = recent_df['atr'].mean()
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1
        
        if roc > 5 and atr_ratio > 1.2:
            score = 0.8
            signal = "Strong Bullish (expanding)"
        elif roc > 2:
            score = 0.4
            signal = "Bullish"
        elif roc < -5 and atr_ratio > 1.2:
            score = -0.8
            signal = "Strong Bearish (expanding)"
        elif roc < -2:
            score = -0.4
            signal = "Bearish"
        else:
            score = 0.0
            signal = "Consolidating"
        
        analysis = f"Momentum {signal} (ROC: {roc:.1f}%, ATR: {atr_ratio:.2f}x)"
        return score, analysis


def analyze_crypto_pair(ticker: str, timeframe: str = '1h', 
                        lookback_days: int = 30,
                        exchange: str = 'binance') -> CryptoSignal:
    """
    Analyze a cryptocurrency pair and generate signal
    
    Args:
        ticker: Trading pair (e.g., 'BTC/USDT')
        timeframe: Analysis timeframe
        lookback_days: Days of historical data
        exchange: Exchange name
        
    Returns:
        CryptoSignal with analysis
    """
    # Fetch data
    dm = DataManager(exchange)
    df = dm.fetch_ohlcv(ticker, timeframe, lookback_days)
    weekly_df = dm.fetch_ohlcv(ticker, '1w', lookback_days * 2)
    
    # Calculate indicators
    df = TechnicalIndicators.add_all_indicators(df, weekly_df)
    
    # Calculate key levels
    market_profile = KeyLevels.market_profile(df)
    df['poc'] = market_profile['poc']
    df['vah'] = market_profile['vah']
    df['val'] = market_profile['val']
    
    # Analyze
    analyzer = CryptoSignalAnalyzer()
    signal = analyzer.analyze(ticker, df, weekly_df)
    
    return signal

