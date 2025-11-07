"""
Enhanced Trading Signal System
Combines stock analysis with cryptocurrency order flow analysis
"""

from typing import List, Tuple
from datetime import datetime, timedelta
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

# Import backtesting modules
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.metrics import MetricsCalculator
from backtesting.visualizer import BacktestVisualizer
from backtesting.reports import BacktestReporter
from backtesting.monte_carlo import MonteCarloSimulator
from backtesting.walk_forward import WalkForwardAnalyzer


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


def backtest_strategy(tickers: List[str], 
                     start_date: str = "2022-01-01",
                     end_date: str = "2024-12-31",
                     config_path: str = "config/weights.yaml",
                     run_monte_carlo: bool = False,
                     run_walk_forward: bool = False):
    """
    Backtest trading strategy on historical data
    
    Args:
        tickers: List of tickers to backtest
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        config_path: Path to configuration file
        run_monte_carlo: Whether to run Monte Carlo simulation
        run_walk_forward: Whether to run walk-forward analysis
    """
    print("\n" + "="*70)
    print("STRATEGY BACKTESTING")
    print("="*70)
    
    # Load configuration
    config = ScoringConfig.from_yaml(config_path)
    
    # Create backtest config from settings
    backtest_config = BacktestConfig(
        initial_capital=config.backtesting.initial_capital,
        commission=config.backtesting.commission,
        slippage=config.backtesting.slippage,
        position_size=config.backtesting.position_size,
        max_positions=config.backtesting.max_positions,
        stop_loss_atr=config.backtesting.stop_loss_atr,
        take_profit_atr=config.backtesting.take_profit_atr,
        holding_period_days=config.backtesting.holding_period_days
    )
    
    # Initialize components
    signal_analyzer = SignalAnalyzer(config)
    _setup_strategies_for_analyzer(signal_analyzer, config)
    
    backtest_engine = BacktestEngine(signal_analyzer, backtest_config)
    visualizer = BacktestVisualizer()
    reporter = BacktestReporter()
    
    # Run backtests for each ticker
    all_results = []
    
    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {ticker}")
        print(f"{'='*70}")
        
        try:
            # Run backtest
            result = backtest_engine.run(ticker, start_date, end_date)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate(result)
            
            # Print summary
            reporter.print_summary(result, metrics)
            
            # Generate visualizations
            visualizer.create_full_report(result, metrics, ticker)
            
            # Export results
            reporter.export_trades_csv(result)
            reporter.export_metrics_json(metrics, result)
            reporter.generate_html_report(result, metrics)
            
            all_results.append((ticker, result, metrics))
            
            # Monte Carlo simulation
            if run_monte_carlo and len(result.closed_trades) > 5:
                print(f"\n[*] Running Monte Carlo simulation for {ticker}...")
                mc_simulator = MonteCarloSimulator(
                    num_simulations=config.monte_carlo.num_simulations,
                    confidence_levels=config.monte_carlo.confidence_levels
                )
                mc_result = mc_simulator.run(result)
                
                # Plot Monte Carlo results
                percentiles = {
                    5: mc_result.percentile_5,
                    50: mc_result.percentile_50,
                    95: mc_result.percentile_95
                }
                visualizer.plot_monte_carlo(
                    mc_result.equity_curves[:100],  # Sample for visualization
                    percentiles=percentiles,
                    save_path=f"results/backtesting/{ticker}_monte_carlo.png"
                )
            
            # Walk-forward analysis
            if run_walk_forward:
                print(f"\n[*] Running walk-forward analysis for {ticker}...")
                wf_analyzer = WalkForwardAnalyzer(signal_analyzer, backtest_config)
                wf_result = wf_analyzer.run(
                    ticker, start_date, end_date,
                    train_days=config.walk_forward.train_days,
                    test_days=config.walk_forward.test_days
                )
                
                # Export and visualize
                wf_analyzer.export_windows_csv(wf_result, 
                    f"results/backtesting/{ticker}_walk_forward.csv")
                wf_analyzer.plot_walk_forward_results(wf_result)
            
        except Exception as e:
            print(f"[ERROR] Backtest failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Strategy comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print(f"{'='*70}")
        
        comparison_df = reporter.compare_strategies(all_results)
        print("\n", comparison_df.to_string())
        reporter.export_comparison_csv(comparison_df)
    
    print(f"\n{'='*70}")
    print("BACKTESTING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: results/backtesting/")
    print(f"{'='*70}\n")


def _setup_strategies_for_analyzer(analyzer: SignalAnalyzer, config: ScoringConfig):
    """Helper function to register strategies with analyzer"""
    weights = config.weights
    
    # Momentum strategies
    analyzer.register_strategy(RSIStrategy(weights.rsi))
    analyzer.register_strategy(MACDStrategy(weights.macd))
    analyzer.register_strategy(OBVStrategy(weights.obv))
    
    # Trend strategies
    analyzer.register_strategy(MovingAverageCrossStrategy(weights.ma_crossover))
    
    # Volatility strategies
    analyzer.register_strategy(BollingerBandStrategy(weights.bollinger_bands))
    
    # Volume strategies
    analyzer.register_strategy(VolumeStrategy(weights.volume))
    
    # Pattern strategies
    analyzer.register_strategy(CandlestickPatternStrategy(weights.candlestick))


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


def run_comprehensive_backtest():
    """
    Run backtesting on US stocks, Singapore stocks, and crypto
    Uses the same tickers from main analysis for consistency
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE BACKTESTING")
    print(f"Testing strategies on historical data (2022 - {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')})")
    print("="*70)
    
    # Same tickers as main analysis
    US_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META"]
    SG_TICKERS = ["CRPU.SI", "J69U.SI", "BUOU.SI", "M44U.SI"]
    
    # Backtest date range
    START_DATE = "2022-01-01"
    END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # Yesterday
    
    # ========== US STOCK BACKTESTING ==========
    print("\n" + "="*70)
    print("BACKTESTING US STOCKS")
    print("="*70)
    
    try:
        backtest_strategy(
            tickers=US_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            run_monte_carlo=False,
            run_walk_forward=False
        )
    except Exception as e:
        print(f"[ERROR] US stock backtesting failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== SINGAPORE STOCK BACKTESTING ==========
    print("\n" + "="*70)
    print("BACKTESTING SINGAPORE STOCKS")
    print("="*70)
    
    try:
        backtest_strategy(
            tickers=SG_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            run_monte_carlo=False,
            run_walk_forward=False
        )
    except Exception as e:
        print(f"[ERROR] Singapore stock backtesting failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== CRYPTO BACKTESTING NOTE ==========
    print("\n" + "="*70)
    print("CRYPTO BACKTESTING")
    print("="*70)
    print("\n[INFO] Crypto backtesting uses a different methodology (TraderXO)")
    print("[INFO] Crypto analysis focuses on real-time order flow and momentum")
    print("[INFO] For crypto historical testing, use the TraderXO charts generated")
    print("       which show past signals on price action")
    print("\n[NOTE] Stock-style backtesting is designed for daily/weekly timeframes")
    print("       Crypto strategies work best on 1h/4h timeframes with live data")
    
    print("\n" + "="*70)
    print("BACKTESTING COMPLETE")
    print("="*70)
    print("\n[RESULTS] Results Location:")
    print("   - HTML Reports: results/backtesting/YYYY-MM-DD/")
    print("   - Trade History: results/backtesting/YYYY-MM-DD/*_trades.csv")
    print("   - Charts: results/backtesting/YYYY-MM-DD/*.png")
    print("   - Comparison: results/backtesting/strategy_comparison.csv")
    print("\n[TIP] Tip: Open the HTML reports in your browser for interactive analysis!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run main trading analysis
    main()
    
    # Run comprehensive backtesting (US + SG stocks)
    run_comprehensive_backtest()

