"""
Trading Signal System - Refactored Architecture
Main entry point for running signal analysis
"""

from typing import List, Tuple
from config.settings import ScoringConfig
from analysis.signal_generator import SignalAnalyzer
from analysis.strategies.momentum import RSIStrategy, MACDStrategy, OBVStrategy
from analysis.strategies.trend import MovingAverageCrossStrategy
from analysis.strategies.volatility import BollingerBandStrategy
from analysis.strategies.volume import VolumeStrategy
from technical.patterns.strategy import CandlestickPatternStrategy
from data.providers.seeking_alpha import SeekingAlphaProvider
from data.providers.yfinance import YFinanceProvider
from output.formatters import SignalFormatter
from visualization.chart_generator import ChartGenerator
from core.models import TradingSignal, MarketData
from core.enums import TimeFrame


class TradingSignalSystem:
    """Main trading signal system orchestrator"""
    
    def __init__(self, config_path: str = "config/weights.yaml"):
        """
        Initialize the trading signal system
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = ScoringConfig.from_yaml(config_path)
        
        # Initialize analyzer
        self.analyzer = SignalAnalyzer(self.config)
        
        # Register all strategies
        self._setup_strategies()
        
    def _setup_strategies(self):
        """Register all analysis strategies with their weights"""
        weights = self.config.weights
        
        # Momentum strategies
        self.analyzer.register_strategy(RSIStrategy(weights.rsi))
        self.analyzer.register_strategy(MACDStrategy(weights.macd))
        self.analyzer.register_strategy(OBVStrategy(weights.obv))
        
        # Trend strategies
        self.analyzer.register_strategy(
            MovingAverageCrossStrategy(weights.ma_crossover)
        )
        
        # Volatility strategies
        self.analyzer.register_strategy(
            BollingerBandStrategy(weights.bollinger_bands)
        )
        
        # Volume strategies
        self.analyzer.register_strategy(VolumeStrategy(weights.volume))
        
        # Pattern strategies
        self.analyzer.register_strategy(
            CandlestickPatternStrategy(weights.candlestick)
        )
    
    def analyze_tickers(self, tickers: List[str], 
                       market: str = "US",
                       timeframe: TimeFrame = TimeFrame.DAILY) -> Tuple[List[TradingSignal], List[MarketData]]:
        """
        Analyze list of tickers and generate trading signals
        
        Args:
            tickers: List of ticker symbols
            market: Market type ("US" or "SG")
            timeframe: Analysis timeframe
            
        Returns:
            Tuple of (List of trading signals, List of market data)
        """
        # Choose data provider based on market
        if market == "US":
            provider = SeekingAlphaProvider()
            print(f"\n[*] Using Seeking Alpha for US market data")
            # Seeking Alpha - process one by one (has its own rate limiting)
            return self._analyze_tickers_sequential(provider, tickers, timeframe)
        else:
            provider = YFinanceProvider()
            print(f"\n[*] Using yfinance for {market} market data")
            # yfinance - use batch download to avoid rate limiting
            return self._analyze_tickers_batch(provider, tickers, timeframe)
    
    def _analyze_tickers_sequential(self, provider, tickers: List[str],
                                    timeframe: TimeFrame) -> Tuple[List[TradingSignal], List[MarketData]]:
        """Process tickers one by one (for Seeking Alpha)"""
        signals = []
        market_data_list = []
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker}...", end=" ")
                
                # Fetch and prepare market data
                market_data = provider.get_market_data(
                    ticker, 
                    period="2y",
                    timeframe=timeframe
                )
                
                # Generate signal
                signal = self.analyzer.analyze(market_data)
                signals.append(signal)
                market_data_list.append(market_data)
                
                print(f"[OK] {signal.signal.value}")
                
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        
        return signals, market_data_list
    
    def _analyze_tickers_batch(self, provider, tickers: List[str],
                               timeframe: TimeFrame) -> Tuple[List[TradingSignal], List[MarketData]]:
        """Process tickers in batch (for yfinance to avoid rate limiting)"""
        signals = []
        saved_market_data = []
        
        # Get all market data at once using batch download
        try:
            market_data_list = provider.get_market_data_batch(
                tickers,
                period="2y",
                timeframe=timeframe
            )
            
            # Generate signals for each ticker
            for market_data in market_data_list:
                try:
                    signal = self.analyzer.analyze(market_data)
                    signals.append(signal)
                    saved_market_data.append(market_data)
                except Exception as e:
                    print(f"  [ERROR] Failed to analyze {market_data.ticker}: {e}")
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Batch download failed: {e}")
            print("[*] Falling back to sequential processing...")
            # Fallback to sequential if batch fails
            return self._analyze_tickers_sequential(provider, tickers, timeframe)
        
        return signals, saved_market_data


def main():
    """Main execution function"""
    # Configuration
    US_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
        "BABA", "V", "MA", "AMD", "VOO", "META"
    ]
    
    SG_TICKERS = [
        "CRPU.SI", "J69U.SI", "BUOU.SI", "M44U.SI",
        "ME8U.SI", "JYEU.SI"
    ]
    
    # Initialize system
    print("\n" + "="*70)
    print("TRADING SIGNAL ANALYSIS SYSTEM")
    print("Refactored Architecture with Strategy Pattern")
    print("="*70)
    
    system = TradingSignalSystem()
    chart_generator = ChartGenerator()
    
    # Analyze US Market
    print("\n" + "="*70)
    print("US MARKET ANALYSIS")
    print("="*70)
    
    us_signals, us_market_data = system.analyze_tickers(US_TICKERS, market="US")
    
    if us_signals:
        print(f"\n[SUCCESS] Analyzed {len(us_signals)} US tickers")
        SignalFormatter.print_signals(us_signals, detailed=True)
        
        # Export to CSV and JSON
        try:
            SignalFormatter.export_to_csv(us_signals, "us_signals.csv")
            SignalFormatter.export_to_json(us_signals, "us_signals")
        except Exception as e:
            print(f"[WARNING] Could not export files: {e}")
        
        # Generate charts
        print(f"\n[*] Generating charts for US market tickers...")
        for signal, market_data in zip(us_signals, us_market_data):
            try:
                chart_generator.generate_all_charts(market_data, signal)
            except Exception as e:
                print(f"  [ERROR] Chart generation failed for {signal.ticker}: {e}")
        
        # Generate summary report
        try:
            chart_generator.generate_summary_report(us_signals, "US")
        except Exception as e:
            print(f"[WARNING] Could not generate summary report: {e}")
    else:
        print("\n[WARNING] No valid US signals generated")
    
    # Analyze Singapore Market
    print("\n" + "="*70)
    print("SINGAPORE MARKET ANALYSIS")
    print("="*70)
    
    sg_signals, sg_market_data = system.analyze_tickers(SG_TICKERS, market="SG")
    
    if sg_signals:
        print(f"\n[SUCCESS] Analyzed {len(sg_signals)} Singapore tickers")
        SignalFormatter.print_signals(sg_signals, detailed=True)
        
        # Export to CSV and JSON
        try:
            SignalFormatter.export_to_csv(sg_signals, "sg_signals.csv")
            SignalFormatter.export_to_json(sg_signals, "sg_signals")
        except Exception as e:
            print(f"[WARNING] Could not export files: {e}")
        
        # Generate charts
        print(f"\n[*] Generating charts for Singapore market tickers...")
        for signal, market_data in zip(sg_signals, sg_market_data):
            try:
                chart_generator.generate_all_charts(market_data, signal)
            except Exception as e:
                print(f"  [ERROR] Chart generation failed for {signal.ticker}: {e}")
        
        # Generate summary report
        try:
            chart_generator.generate_summary_report(sg_signals, "SG")
        except Exception as e:
            print(f"[WARNING] Could not generate summary report: {e}")
    else:
        print("\n[WARNING] No valid Singapore signals generated")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total Tickers Analyzed: {len(us_signals) + len(sg_signals)}")
    print(f"  US Market: {len(us_signals)}")
    print(f"  Singapore Market: {len(sg_signals)}")
    
    # Export combined results (CSV and JSON)
    try:
        SignalFormatter.export_combined_to_csv(us_signals, sg_signals)
        SignalFormatter.export_combined_to_json(us_signals, sg_signals)
    except Exception as e:
        print(f"[WARNING] Could not export combined files: {e}")
    
    # Show only actionable signals
    all_signals = us_signals + sg_signals
    actionable = [s for s in all_signals if s.is_actionable]
    
    if actionable:
        print(f"\n[ACTIONABLE] {len(actionable)} signals require action:")
        SignalFormatter.print_actionable_only(all_signals, detailed=False)
    else:
        print("\n[INFO] No actionable signals at this time (all HOLD)")


if __name__ == "__main__":
    main()

