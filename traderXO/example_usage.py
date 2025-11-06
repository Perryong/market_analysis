"""
Example usage of Trader_XO visualization system

This script demonstrates how to use the three main plotting strategies:
1. Order Flow & Market Profile
2. Momentum Strategy
3. Range Fading Strategy
"""

from traderXO.orderflow_profile import plot_orderflow_profile
from traderXO.momentum_strategy import plot_momentum_strategy
from traderXO.range_fade import plot_range_fade


def main():
    """Run all three visualization strategies"""
    
    # Configuration
    TICKER = 'BTC/USDT'  # Trading pair
    EXCHANGE = 'binance'  # Exchange to use
    
    print("\n" + "="*80)
    print("TRADER_XO VISUALIZATION SYSTEM")
    print("="*80)
    
    # ===== STRATEGY 1: Order Flow & Market Profile =====
    print("\n[1/3] Generating Order Flow & Market Profile visualization...")
    try:
        plot_orderflow_profile(
            ticker=TICKER,
            timeframe='1h',      # 1-hour candles
            lookback_days=30,    # Last 30 days
            exchange=EXCHANGE
        )
    except Exception as e:
        print(f"[ERROR] Order Flow plot failed: {e}")
    
    # ===== STRATEGY 2: Momentum Strategy =====
    print("\n[2/3] Generating Momentum Strategy visualization...")
    try:
        plot_momentum_strategy(
            ticker=TICKER,
            timeframe='1d',        # Daily candles
            lookback_days=180,     # Last 6 months
            exchange=EXCHANGE,
            benchmark_ticker='ETH/USDT'  # Compare to ETH for relative strength
        )
    except Exception as e:
        print(f"[ERROR] Momentum plot failed: {e}")
    
    # ===== STRATEGY 3: Range Fading Strategy =====
    print("\n[3/3] Generating Range Fading Strategy visualization...")
    try:
        plot_range_fade(
            ticker=TICKER,
            range_high=None,     # Auto-detect range
            range_low=None,      # Auto-detect range
            timeframe='4h',      # 4-hour candles
            lookback_days=60,    # Last 60 days
            exchange=EXCHANGE
        )
    except Exception as e:
        print(f"[ERROR] Range Fade plot failed: {e}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("Check the traderXO/plots/ directory for generated charts")
    print("="*80 + "\n")


def example_with_custom_ranges():
    """Example with manually specified range for range fading"""
    
    plot_range_fade(
        ticker='ETH/USDT',
        range_high=3500.0,   # Manual range high
        range_low=3000.0,    # Manual range low
        timeframe='4h',
        lookback_days=30,
        exchange='binance'
    )


def example_multiple_tickers():
    """Example: Generate charts for multiple tickers"""
    
    tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MATIC/USDT']
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Processing {ticker}")
        print(f"{'='*80}")
        
        # Order Flow
        try:
            plot_orderflow_profile(ticker, '1h', 30, 'binance')
        except Exception as e:
            print(f"[ERROR] {ticker} orderflow: {e}")
        
        # Momentum
        try:
            plot_momentum_strategy(ticker, '1d', 180, 'binance')
        except Exception as e:
            print(f"[ERROR] {ticker} momentum: {e}")
        
        # Range Fade
        try:
            plot_range_fade(ticker, None, None, '4h', 60, 'binance')
        except Exception as e:
            print(f"[ERROR] {ticker} range: {e}")


def example_intraday_scalping():
    """Example for intraday scalping with 5-minute charts"""
    
    plot_orderflow_profile(
        ticker='BTC/USDT',
        timeframe='5m',      # 5-minute candles
        lookback_days=3,     # Last 3 days
        exchange='binance'
    )


if __name__ == '__main__':
    # Run the main example
    main()
    
    # Uncomment to run other examples:
    # example_with_custom_ranges()
    # example_multiple_tickers()
    # example_intraday_scalping()

