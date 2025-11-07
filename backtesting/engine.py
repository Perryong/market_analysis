"""
Core Backtesting Engine

Simulates trading strategy execution on historical data with realistic
trade execution, position management, and portfolio tracking.
"""

from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

from backtesting.models import (
    Trade, Position, Portfolio, BacktestResult,
    TradeDirection, TradeStatus
)
from core.models import MarketData, TradingSignal
from core.enums import SignalType, TimeFrame
from analysis.signal_generator import SignalAnalyzer
from data.providers.yfinance import YFinanceProvider


class BacktestConfig:
    """Backtesting configuration"""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.0005,   # 0.05%
                 position_size: float = 0.1,  # 10% of equity per trade
                 max_positions: int = 10,
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 3.0,
                 holding_period_days: Optional[int] = None):
        """
        Initialize backtest configuration
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            position_size: Position size as fraction of equity
            max_positions: Maximum concurrent positions
            stop_loss_atr: Stop loss distance in ATR multiples
            take_profit_atr: Take profit distance in ATR multiples
            holding_period_days: Maximum holding period (None = unlimited)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.holding_period_days = holding_period_days


class BacktestEngine:
    """
    Core backtesting engine
    
    Simulates strategy execution on historical data with realistic
    trade costs, position management, and portfolio tracking.
    """
    
    def __init__(self, 
                 signal_analyzer: SignalAnalyzer,
                 config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine
        
        Args:
            signal_analyzer: Signal generator for trading decisions
            config: Backtesting configuration (uses defaults if None)
        """
        self.signal_analyzer = signal_analyzer
        self.config = config or BacktestConfig()
        self.data_provider = YFinanceProvider()
        
    def run(self,
            ticker: str,
            start_date: str,
            end_date: str,
            timeframe: TimeFrame = TimeFrame.DAILY) -> BacktestResult:
        """
        Run backtest for a single ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            BacktestResult with all trades and portfolio snapshots
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {ticker}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"{'='*70}")
        
        # Fetch historical data
        print(f"[*] Fetching historical data...")
        df = self._fetch_data(ticker, start_date, end_date, timeframe)
        
        if df is None or len(df) < 50:
            print(f"[ERROR] Insufficient data for {ticker}")
            return self._create_empty_result(ticker, start_date, end_date)
        
        print(f"[*] Loaded {len(df)} data points")
        
        # Initialize portfolio
        portfolio = Portfolio(
            date=df.index[0],
            cash=self.config.initial_capital,
            positions=[]
        )
        
        # Tracking
        all_trades: List[Trade] = []
        all_snapshots: List[Portfolio] = []
        equity_curve_data = []
        
        # Simulate day by day
        print(f"[*] Running simulation...")
        
        for i in range(len(df)):
            current_date = df.index[i]
            
            # Update positions with current prices
            self._update_positions(portfolio, df.iloc[i], current_date)
            
            # Check for exit conditions (stop loss, take profit, time)
            closed_trades = self._check_exit_conditions(
                portfolio, df.iloc[i], current_date, all_trades
            )
            all_trades.extend(closed_trades)
            
            # Generate signal for new entry (if conditions met)
            if self._can_open_position(portfolio):
                try:
                    # Create market data for signal generation
                    # Need sufficient history
                    if i >= 50:
                        market_data = self._create_market_data(
                            ticker, df.iloc[:i+1], timeframe
                        )
                        signal = self.signal_analyzer.analyze(market_data)
                        
                        # Process signal
                        new_trade = self._process_signal(
                            signal, portfolio, df.iloc[i], current_date
                        )
                        if new_trade:
                            all_trades.append(new_trade)
                except Exception as e:
                    # Log signal generation errors (but continue)
                    if i == 50:  # Only print once to avoid spam
                        print(f"[WARNING] Signal generation error: {e}")
                    pass
            
            # Record portfolio snapshot
            snapshot = deepcopy(portfolio)
            all_snapshots.append(snapshot)
            equity_curve_data.append({
                'date': current_date,
                'equity': portfolio.equity
            })
        
        # Close any remaining open positions at end
        final_date = df.index[-1]
        final_price_data = df.iloc[-1]
        remaining_trades = self._close_all_positions(
            portfolio, final_price_data, final_date, "End of backtest"
        )
        all_trades.extend(remaining_trades)
        
        # Create equity curve
        equity_df = pd.DataFrame(equity_curve_data)
        equity_curve = equity_df.set_index('date')['equity']
        
        # Build result
        result = BacktestResult(
            ticker=ticker,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            initial_capital=self.config.initial_capital,
            trades=all_trades,
            equity_curve=equity_curve,
            portfolio_snapshots=all_snapshots,
            commission=self.config.commission,
            slippage=self.config.slippage,
            position_size=self.config.position_size,
            strategy_name="Technical Analysis Strategy",
            strategy_params={}
        )
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _fetch_data(self, ticker: str, start_date: str, end_date: str,
                    timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        """Fetch historical data for backtesting (uses cached data when available)"""
        import time
        
        try:
            # Calculate period to fetch (need extra for indicators)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # Get extra 200 days for indicator calculation
            fetch_start = start_dt - timedelta(days=200)
            
            # Try to load from cache first (yfinance cache)
            cache_dir = Path(".cache/yfinance")
            cache_file = cache_dir / f"{ticker}_2y_1day_yf.json"
            
            if cache_file.exists():
                try:
                    print(f"[CACHE] Loading {ticker} from yfinance cache...")
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        df = pd.DataFrame(
                            data['data'],
                            index=pd.to_datetime(data['index']),
                            columns=data['columns']
                        )
                        df.index.name = 'Date'
                        
                        # Filter to needed date range
                        df = df[df.index >= fetch_start]
                        
                        if len(df) > 0:
                            print(f"[CACHE] Loaded {len(df)} rows from cache")
                            return df
                except Exception as e:
                    print(f"[WARNING] Failed to load from cache: {e}")
            
            # Try Seeking Alpha cache for US stocks (if no .SI suffix)
            if not ticker.endswith('.SI'):
                sa_cache = Path(".cache") / f"{ticker}_chart_1y_sa.json"
                if sa_cache.exists():
                    try:
                        print(f"[CACHE] Loading {ticker} from Seeking Alpha cache...")
                        with open(sa_cache, 'r') as f:
                            data = json.load(f)
                            
                        if 'attributes' in data:
                            # Convert SA format to DataFrame
                            records = []
                            for date_str, values in data['attributes'].items():
                                record = {'Date': date_str}
                                record.update(values)
                                records.append(record)
                            
                            df = pd.DataFrame(records)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                            df.sort_index(inplace=True)
                            
                            # Rename columns to match expected format
                            if 'open' in df.columns:
                                df.rename(columns={
                                    'open': 'Open',
                                    'high': 'High',
                                    'low': 'Low',
                                    'close': 'Close',
                                    'volume': 'Volume'
                                }, inplace=True)
                            
                            # Filter to needed date range
                            df = df[df.index >= fetch_start]
                            
                            if len(df) > 0:
                                print(f"[CACHE] Loaded {len(df)} rows from SA cache")
                                return df
                    except Exception as e:
                        print(f"[WARNING] Failed to load from SA cache: {e}")
            
            # If no cache, use YFinanceProvider which has its own caching logic
            print(f"[DOWNLOAD] No cache found, downloading {ticker}...")
            time.sleep(0.5)  # Rate limit protection
            
            df = self.data_provider.fetch_data(ticker, period="2y", timeframe=timeframe)
            
            if df is None or len(df) == 0:
                return None
            
            # Filter to needed date range
            df = df[df.index >= fetch_start]
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return None
    
    def _create_market_data(self, ticker: str, df: pd.DataFrame,
                           timeframe: TimeFrame) -> MarketData:
        """Create MarketData object from DataFrame with technical indicators"""
        # Calculate technical indicators if not already present
        if 'rsi' not in df.columns:
            from technical.calculator import TechnicalCalculator
            calculator = TechnicalCalculator()
            df = calculator.calculate_all(df)
        
        return MarketData(
            ticker=ticker,
            current=df.iloc[-1],
            previous=df.iloc[-2] if len(df) > 1 else df.iloc[-1],
            historical=df,
            timeframe=timeframe,
            short_name=ticker
        )
    
    def _update_positions(self, portfolio: Portfolio,
                         current_data: pd.Series, date: datetime):
        """Update all position prices"""
        current_price = float(current_data['Close'])
        for pos in portfolio.positions:
            pos.update_price(current_price)
    
    def _check_exit_conditions(self, portfolio: Portfolio,
                               current_data: pd.Series,
                               date: datetime,
                               all_trades: List[Trade]) -> List[Trade]:
        """Check and execute exit conditions for open positions"""
        closed_trades = []
        positions_to_remove = []
        
        for pos in portfolio.positions[:]:  # Copy list to allow modification
            exit_reason = None
            
            # Check stop loss
            if pos.should_stop_loss():
                exit_reason = "Stop Loss"
            
            # Check take profit
            elif pos.should_take_profit():
                exit_reason = "Take Profit"
            
            # Check holding period
            elif self.config.holding_period_days:
                days_held = (date - pos.entry_date).days
                if days_held >= self.config.holding_period_days:
                    exit_reason = "Holding Period Expired"
            
            # Execute exit if needed
            if exit_reason:
                trade = self._close_position(
                    pos, portfolio, current_data, date, exit_reason
                )
                closed_trades.append(trade)
                positions_to_remove.append(pos)
        
        # Remove closed positions
        for pos in positions_to_remove:
            portfolio.positions.remove(pos)
        
        return closed_trades
    
    def _can_open_position(self, portfolio: Portfolio) -> bool:
        """Check if we can open a new position"""
        return len(portfolio.positions) < self.config.max_positions
    
    def _process_signal(self, signal: TradingSignal, portfolio: Portfolio,
                       current_data: pd.Series, date: datetime) -> Optional[Trade]:
        """Process trading signal and open position if appropriate"""
        # Only act on BUY/SELL signals (not HOLD)
        if signal.signal == SignalType.HOLD:
            return None
        
        # Check if we already have a position in this ticker
        if portfolio.has_position(signal.ticker):
            return None
        
        # For now, only handle LONG positions (BUY signals)
        if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            return self._open_long_position(
                signal, portfolio, current_data, date
            )
        
        return None
    
    def _open_long_position(self, signal: TradingSignal, portfolio: Portfolio,
                           current_data: pd.Series, date: datetime) -> Trade:
        """Open a long position"""
        price = float(current_data['Close'])
        atr = float(current_data.get('atr', price * 0.02))  # Fallback to 2%
        
        # Calculate position size
        position_value = portfolio.equity * self.config.position_size
        shares = position_value / price
        
        # Calculate costs
        entry_slippage_per_share = price * self.config.slippage
        entry_value = shares * price
        entry_commission = entry_value * self.config.commission
        
        # Check if we have enough cash
        total_cost = entry_value + entry_commission
        if total_cost > portfolio.cash:
            shares = (portfolio.cash * 0.99) / (price + entry_commission/shares)
            entry_value = shares * price
            entry_commission = entry_value * self.config.commission
            total_cost = entry_value + entry_commission
        
        # Calculate stop loss and take profit
        stop_loss = price - (self.config.stop_loss_atr * atr)
        take_profit = price + (self.config.take_profit_atr * atr)
        
        # Create trade
        trade = Trade(
            ticker=signal.ticker,
            direction=TradeDirection.LONG,
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_commission=entry_commission,
            entry_slippage=entry_slippage_per_share,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=TradeStatus.OPEN,
            signal_confidence=signal.confidence,
            entry_reason=f"Signal: {signal.signal.value} ({signal.confidence_percent:.1f}%)"
        )
        
        # Create position
        position = Position(
            ticker=signal.ticker,
            direction=TradeDirection.LONG,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trade=trade
        )
        
        # Update portfolio
        portfolio.cash -= total_cost
        portfolio.positions.append(position)
        
        return trade
    
    def _close_position(self, position: Position, portfolio: Portfolio,
                       current_data: pd.Series, date: datetime,
                       reason: str) -> Trade:
        """Close a position and update trade"""
        exit_price = float(current_data['Close'])
        
        # Get associated trade
        trade = position.trade
        if trade is None:
            raise ValueError("Position has no associated trade")
        
        # Calculate exit costs
        exit_slippage_per_share = exit_price * self.config.slippage
        exit_value = position.shares * exit_price
        exit_commission = exit_value * self.config.commission
        
        # Update trade with exit information
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_commission = exit_commission
        trade.exit_slippage = exit_slippage_per_share
        trade.status = TradeStatus.CLOSED
        trade.exit_reason = reason
        
        # Update portfolio cash
        proceeds = exit_value - exit_commission
        portfolio.cash += proceeds
        
        return trade
    
    def _close_all_positions(self, portfolio: Portfolio,
                            current_data: pd.Series, date: datetime,
                            reason: str) -> List[Trade]:
        """Close all open positions"""
        closed_trades = []
        
        for pos in portfolio.positions[:]:
            trade = self._close_position(pos, portfolio, current_data, date, reason)
            closed_trades.append(trade)
        
        portfolio.positions.clear()
        return closed_trades
    
    def _create_empty_result(self, ticker: str, start_date: str,
                            end_date: str) -> BacktestResult:
        """Create empty result for failed backtest"""
        return BacktestResult(
            ticker=ticker,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            initial_capital=self.config.initial_capital,
            trades=[],
            equity_curve=pd.Series(),
            portfolio_snapshots=[],
            commission=self.config.commission,
            slippage=self.config.slippage,
            position_size=self.config.position_size
        )
    
    def _print_summary(self, result: BacktestResult):
        """Print backtest summary"""
        print(f"\n{'='*70}")
        print("BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total Trades: {result.num_trades}")
        print(f"  Closed: {len(result.closed_trades)}")
        print(f"  Open: {len(result.open_trades)}")
        print(f"  Winners: {len(result.winning_trades)}")
        print(f"  Losers: {len(result.losing_trades)}")
        print(f"\nFinal Equity: ${result.final_equity:,.2f}")
        print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)")
        
        if len(result.closed_trades) > 0:
            win_rate = len(result.winning_trades) / len(result.closed_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        print(f"{'='*70}\n")

