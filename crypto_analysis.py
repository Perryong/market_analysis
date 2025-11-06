"""
Enhanced Trading Signal System
Combines stock analysis with cryptocurrency order flow analysis
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

# Import traderXO modules
from traderXO.orderflow_profile import plot_orderflow_profile
from traderXO.momentum_strategy import plot_momentum_strategy
from traderXO.range_fade import plot_range_fade
from traderXO.crypto_signal_analyzer import analyze_crypto_pair
from traderXO.crypto_signal_formatter import CryptoSignalFormatter
from traderXO.stock_adapter import analyze_stock_with_traderxo, get_stock_data_for_traderxo


class TradingSignalSystem:
    """Main trading signal system orchestrator"""
    
    def __init__(self, config_path: str = "config/weights.yaml"):
        """Initialize the trading signal system"""
        self.config = ScoringConfig.from_yaml(config_path)
        self.analyzer = SignalAnalyzer(self.config)
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
        """Analyze list of tickers and generate trading signals"""
        # Choose data provider based on market
        if market == "US":
            provider = SeekingAlphaProvider()
            print(f"\n[*] Using Seeking Alpha for US market data")
            return self._analyze_tickers_sequential(provider, tickers, timeframe)
        else:
            provider = YFinanceProvider()
            print(f"\n[*] Using yfinance for {market} market data")
            return self._analyze_tickers_batch(provider, tickers, timeframe)
    
    def _analyze_tickers_sequential(self, provider, tickers: List[str],
                                    timeframe: TimeFrame) -> Tuple[List[TradingSignal], List[MarketData]]:
        """Process tickers one by one (for Seeking Alpha)"""
        signals = []
        market_data_list = []
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker}...", end=" ")
                market_data = provider.get_market_data(ticker, period="2y", timeframe=timeframe)
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
        """Process tickers in batch (for yfinance)"""
        signals = []
        saved_market_data = []
        
        try:
            market_data_list = provider.get_market_data_batch(tickers, period="2y", timeframe=timeframe)
            
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
            return self._analyze_tickers_sequential(provider, tickers, timeframe)
        
        return signals, saved_market_data


def analyze_crypto_with_traderxo(crypto_pairs: List[str], exchange: str = 'binance',
                                generate_charts: bool = True):
    """
    Analyze cryptocurrencies using Trader_XO strategies with signal scoring
    
    Args:
        crypto_pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        exchange: Exchange to use (default: binance)
        generate_charts: Whether to generate strategy charts (default: True)
        
    Returns:
        List of CryptoSignals
    """
    print("\n" + "="*70)
    print("CRYPTOCURRENCY ANALYSIS (TRADER_XO)")
    print("="*70)
    
    crypto_signals = []
    
    # Phase 1: Generate Signals
    print("\n[PHASE 1] Generating Trading Signals...")
    for pair in crypto_pairs:
        print(f"  Analyzing {pair}...", end=" ")
        
        try:
            signal = analyze_crypto_pair(
                ticker=pair,
                timeframe='1h',
                lookback_days=30,
                exchange=exchange
            )
            crypto_signals.append(signal)
            print(f"[OK] {signal.signal} ({signal.confidence:.1f}%)")
            
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # Display signals
    if crypto_signals:
        CryptoSignalFormatter.print_signals(crypto_signals, detailed=True)
        
        # Export to CSV
        try:
            CryptoSignalFormatter.export_to_csv(crypto_signals, 'crypto_signals.csv')
        except Exception as e:
            print(f"[WARNING] Could not export crypto signals: {e}")
    
    # Phase 2: Generate Charts (optional)
    if generate_charts and crypto_signals:
        print("\n[PHASE 2] Generating Strategy Charts...")
        for pair in crypto_pairs:
            print(f"\n[*] Creating charts for {pair}...")
            
            try:
                # Strategy 1: Order Flow & Market Profile
                print(f"  [1/3] Order Flow & Market Profile...")
                plot_orderflow_profile(
                    ticker=pair,
                    timeframe='1h',
                    lookback_days=30,
                    exchange=exchange
                )
                
                # Strategy 2: Momentum
                print(f"  [2/3] Momentum Strategy...")
                plot_momentum_strategy(
                    ticker=pair,
                    timeframe='1d',
                    lookback_days=180,
                    exchange=exchange
                )
                
                # Strategy 3: Range Fade
                print(f"  [3/3] Range Fading Strategy...")
                plot_range_fade(
                    ticker=pair,
                    range_high=None,  # Auto-detect
                    range_low=None,   # Auto-detect
                    timeframe='4h',
                    lookback_days=60,
                    exchange=exchange
                )
                
                print(f"  [SUCCESS] {pair} charts complete")
                
            except Exception as e:
                print(f"  [ERROR] Failed to generate charts for {pair}: {e}")
    
    return crypto_signals


def main():
    """Main execution function"""
    
    # ========== STOCK MARKET CONFIGURATION ==========
    US_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
        "BABA", "V", "MA", "AMD", "VOO", "META"
    ]
    
    SG_TICKERS = [
        "CRPU.SI", "J69U.SI", "BUOU.SI", "M44U.SI",
        "ME8U.SI", "JYEU.SI"
    ]
    
    # ========== CRYPTO CONFIGURATION ==========
    CRYPTO_PAIRS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT" 
    ]
    EXCHANGE = "binance"
    
    # Initialize system
    print("\n" + "="*70)
    print("COMPREHENSIVE TRADING ANALYSIS SYSTEM")
    print("Stock Market + Cryptocurrency Order Flow")
    print("="*70)
    
    # ========== PART 1: STOCK MARKET ANALYSIS ==========
    system = TradingSignalSystem()
    chart_generator = ChartGenerator()
    
    # Analyze US Market
    print("\n" + "="*70)
    print("US STOCK MARKET ANALYSIS")
    print("="*70)
    
    us_signals, us_market_data = system.analyze_tickers(US_TICKERS, market="US")
    
    # Apply TraderXO analysis to US stocks
    us_traderxo_signals = []
    if us_market_data:
        print(f"\n[*] Applying TraderXO analysis to US stocks...")
        for market_data in us_market_data:
            try:
                print(f"  TraderXO analyzing {market_data.ticker}...", end=" ")
                df, weekly_df = get_stock_data_for_traderxo(market_data)
                traderxo_signal = analyze_stock_with_traderxo(market_data.ticker, df, weekly_df)
                us_traderxo_signals.append(traderxo_signal)
                print(f"[OK] {traderxo_signal.signal} ({traderxo_signal.confidence:.1f}%)")
            except Exception as e:
                print(f"[ERROR] {e}")
    
    if us_signals:
        print(f"\n[SUCCESS] Analyzed {len(us_signals)} US tickers")
        print(f"\n{'='*70}")
        print("TRADITIONAL STOCK ANALYSIS")
        print(f"{'='*70}")
        SignalFormatter.print_signals(us_signals, detailed=True)
        
        # Export results
        try:
            SignalFormatter.export_to_csv(us_signals, "us_signals.csv")
            SignalFormatter.export_to_json(us_signals, "us_signals")
        except Exception as e:
            print(f"[WARNING] Could not export files: {e}")
        
        # Show TraderXO signals
        if us_traderxo_signals:
            print(f"\n{'='*70}")
            print("TRADER_XO ANALYSIS (US STOCKS)")
            print(f"{'='*70}")
            CryptoSignalFormatter.print_signals(us_traderxo_signals, detailed=True)
            try:
                CryptoSignalFormatter.export_to_csv(us_traderxo_signals, 'us_traderxo_signals.csv')
            except Exception as e:
                print(f"[WARNING] Could not export TraderXO signals: {e}")
        
        # Generate stock charts
        print(f"\n[*] Generating charts for US market tickers...")
        for signal, market_data in zip(us_signals, us_market_data):
            try:
                chart_generator.generate_all_charts(market_data, signal)
            except Exception as e:
                print(f"  [ERROR] Chart generation failed for {signal.ticker}: {e}")
        
        try:
            chart_generator.generate_summary_report(us_signals, "US")
        except Exception as e:
            print(f"[WARNING] Could not generate summary report: {e}")
    else:
        print("\n[WARNING] No valid US signals generated")
    
    # Analyze Singapore Market
    print("\n" + "="*70)
    print("SINGAPORE STOCK MARKET ANALYSIS")
    print("="*70)
    
    sg_signals, sg_market_data = system.analyze_tickers(SG_TICKERS, market="SG")
    
    # Apply TraderXO analysis to SG stocks
    sg_traderxo_signals = []
    if sg_market_data:
        print(f"\n[*] Applying TraderXO analysis to Singapore stocks...")
        for market_data in sg_market_data:
            try:
                print(f"  TraderXO analyzing {market_data.ticker}...", end=" ")
                df, weekly_df = get_stock_data_for_traderxo(market_data)
                traderxo_signal = analyze_stock_with_traderxo(market_data.ticker, df, weekly_df)
                sg_traderxo_signals.append(traderxo_signal)
                print(f"[OK] {traderxo_signal.signal} ({traderxo_signal.confidence:.1f}%)")
            except Exception as e:
                print(f"[ERROR] {e}")
    
    if sg_signals:
        print(f"\n[SUCCESS] Analyzed {len(sg_signals)} Singapore tickers")
        print(f"\n{'='*70}")
        print("TRADITIONAL STOCK ANALYSIS")
        print(f"{'='*70}")
        SignalFormatter.print_signals(sg_signals, detailed=True)
        
        # Export results
        try:
            SignalFormatter.export_to_csv(sg_signals, "sg_signals.csv")
            SignalFormatter.export_to_json(sg_signals, "sg_signals")
        except Exception as e:
            print(f"[WARNING] Could not export files: {e}")
        
        # Show TraderXO signals
        if sg_traderxo_signals:
            print(f"\n{'='*70}")
            print("TRADER_XO ANALYSIS (SG STOCKS)")
            print(f"{'='*70}")
            CryptoSignalFormatter.print_signals(sg_traderxo_signals, detailed=True)
            try:
                CryptoSignalFormatter.export_to_csv(sg_traderxo_signals, 'sg_traderxo_signals.csv')
            except Exception as e:
                print(f"[WARNING] Could not export TraderXO signals: {e}")
        
        # Generate stock charts
        print(f"\n[*] Generating charts for Singapore market tickers...")
        for signal, market_data in zip(sg_signals, sg_market_data):
            try:
                chart_generator.generate_all_charts(market_data, signal)
            except Exception as e:
                print(f"  [ERROR] Chart generation failed for {signal.ticker}: {e}")
        
        try:
            chart_generator.generate_summary_report(sg_signals, "SG")
        except Exception as e:
            print(f"[WARNING] Could not generate summary report: {e}")
    else:
        print("\n[WARNING] No valid Singapore signals generated")
    
    # Stock Market Summary
    print("\n" + "="*70)
    print("STOCK MARKET ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total Stock Tickers Analyzed: {len(us_signals) + len(sg_signals)}")
    print(f"  US Market: {len(us_signals)}")
    print(f"  Singapore Market: {len(sg_signals)}")
    
    # Export combined results
    try:
        SignalFormatter.export_combined_to_csv(us_signals, sg_signals)
        SignalFormatter.export_combined_to_json(us_signals, sg_signals)
    except Exception as e:
        print(f"[WARNING] Could not export combined files: {e}")
    
    # Show actionable signals
    all_signals = us_signals + sg_signals
    actionable = [s for s in all_signals if s.is_actionable]
    
    if actionable:
        print(f"\n[ACTIONABLE] {len(actionable)} stock signals require action:")
        SignalFormatter.print_actionable_only(all_signals, detailed=False)
    else:
        print("\n[INFO] No actionable stock signals at this time (all HOLD)")
    
    # ========== PART 2: CRYPTOCURRENCY ANALYSIS ==========
    crypto_signals = analyze_crypto_with_traderxo(CRYPTO_PAIRS, EXCHANGE, generate_charts=True)
    
    # Show actionable crypto signals
    if crypto_signals:
        actionable_crypto = [s for s in crypto_signals if s.signal in ['BUY', 'SELL']]
        if actionable_crypto:
            print(f"\n[ACTIONABLE] {len(actionable_crypto)} crypto signals require action:")
            CryptoSignalFormatter.print_actionable_only(crypto_signals, detailed=False)
        else:
            print("\n[INFO] No actionable crypto signals at this time (all HOLD)")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    
    # Stock summary
    print(f"\nStock Tickers Analyzed: {len(us_signals) + len(sg_signals)}")
    print(f"  - US Market: {len(us_signals)}")
    print(f"  - Singapore Market: {len(sg_signals)}")
    if actionable:
        print(f"  - Actionable Signals (Traditional): {len(actionable)}")
    
    # TraderXO stock summary
    all_stock_traderxo = us_traderxo_signals + sg_traderxo_signals
    if all_stock_traderxo:
        stock_traderxo_summary = CryptoSignalFormatter.get_signal_summary(all_stock_traderxo)
        print(f"\nTraderXO Stock Analysis:")
        print(f"  - Total: {stock_traderxo_summary['total']}")
        print(f"  - BUY: {stock_traderxo_summary['buy']}")
        print(f"  - SELL: {stock_traderxo_summary['sell']}")
        print(f"  - HOLD: {stock_traderxo_summary['hold']}")
        print(f"  - Avg Confidence: {stock_traderxo_summary['avg_confidence']:.1f}%")
        print(f"  - Actionable Signals (TraderXO): {stock_traderxo_summary['actionable']}")
    
    # Crypto summary
    if crypto_signals:
        crypto_summary = CryptoSignalFormatter.get_signal_summary(crypto_signals)
        print(f"\nCrypto Pairs Analyzed: {crypto_summary['total']}")
        print(f"  - BUY Signals: {crypto_summary['buy']}")
        print(f"  - SELL Signals: {crypto_summary['sell']}")
        print(f"  - HOLD Signals: {crypto_summary['hold']}")
        print(f"  - Avg Confidence: {crypto_summary['avg_confidence']:.1f}%")
        print(f"  - Actionable Signals: {crypto_summary['actionable']}")
    
    print("\nOutput Locations:")
    print(f"  - Stock Results (Traditional): results/")
    print(f"  - Stock Results (TraderXO): traderXO/results/")
    print(f"  - Stock Charts: plots/YYYY-MM-DD/")
    print(f"  - Crypto Results: traderXO/results/")
    print(f"  - Crypto Charts: traderXO/plots/")
    print("="*70 + "\n")
    
    # Comparison Analysis
    print("\n" + "="*70)
    print("SIGNAL COMPARISON: TRADITIONAL vs TRADER_XO")
    print("="*70)
    
    if us_signals and us_traderxo_signals:
        print("\nUS STOCKS:")
        for trad_sig, txo_sig in zip(us_signals, us_traderxo_signals):
            agreement = "✓ AGREE" if trad_sig.signal.value == txo_sig.signal else "✗ DIFFER"
            print(f"  {trad_sig.ticker:6s} | Traditional: {trad_sig.signal.value:4s} ({trad_sig.confidence_percent:5.1f}%) | "
                  f"TraderXO: {txo_sig.signal:4s} ({txo_sig.confidence:5.1f}%) | {agreement}")
    
    if sg_signals and sg_traderxo_signals:
        print("\nSINGAPORE STOCKS:")
        for trad_sig, txo_sig in zip(sg_signals, sg_traderxo_signals):
            agreement = "✓ AGREE" if trad_sig.signal.value == txo_sig.signal else "✗ DIFFER"
            print(f"  {trad_sig.ticker:10s} | Traditional: {trad_sig.signal.value:4s} ({trad_sig.confidence_percent:5.1f}%) | "
                  f"TraderXO: {txo_sig.signal:4s} ({txo_sig.confidence:5.1f}%) | {agreement}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

