"""
Data models for backtesting framework

Defines core data structures for trades, positions, portfolios, and backtest results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """
    Individual trade record
    
    Tracks entry, exit, P&L, and all trade-related information
    """
    ticker: str
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    shares: float
    
    # Exit information (None if still open)
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    # Trade costs
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    
    # Trade management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    
    # Signal information
    signal_confidence: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.status == TradeStatus.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if trade is closed"""
        return self.status == TradeStatus.CLOSED
    
    @property
    def entry_value(self) -> float:
        """Total entry value including costs"""
        actual_price = self.entry_price + self.entry_slippage
        return (self.shares * actual_price) + self.entry_commission
    
    @property
    def exit_value(self) -> Optional[float]:
        """Total exit value including costs (None if not closed)"""
        if not self.is_closed or self.exit_price is None:
            return None
        
        actual_price = self.exit_price - self.exit_slippage
        return (self.shares * actual_price) - self.exit_commission
    
    @property
    def pnl(self) -> Optional[float]:
        """Profit/Loss in dollars (None if not closed)"""
        if not self.is_closed or self.exit_value is None:
            return None
        
        if self.direction == TradeDirection.LONG:
            return self.exit_value - self.entry_value
        else:  # SHORT
            return self.entry_value - self.exit_value
    
    @property
    def pnl_percent(self) -> Optional[float]:
        """Profit/Loss as percentage (None if not closed)"""
        if self.pnl is None:
            return None
        return (self.pnl / self.entry_value) * 100
    
    @property
    def duration_days(self) -> Optional[int]:
        """Trade duration in days (None if not closed)"""
        if not self.is_closed or self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days
    
    @property
    def mae(self) -> float:
        """Maximum Adverse Excursion (will be calculated by engine)"""
        return 0.0  # Placeholder - calculated during backtesting
    
    @property
    def mfe(self) -> float:
        """Maximum Favorable Excursion (will be calculated by engine)"""
        return 0.0  # Placeholder - calculated during backtesting
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for export"""
        return {
            'ticker': self.ticker,
            'direction': self.direction.value,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'shares': self.shares,
            'exit_date': self.exit_date.strftime('%Y-%m-%d') if self.exit_date else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'duration_days': self.duration_days,
            'entry_commission': self.entry_commission,
            'exit_commission': self.exit_commission,
            'status': self.status.value,
            'signal_confidence': self.signal_confidence,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
        }


@dataclass
class Position:
    """
    Open position tracking
    
    Represents a currently held position in the portfolio
    """
    ticker: str
    direction: TradeDirection
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Associated trade reference
    trade: Optional[Trade] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Cost basis of position"""
        return self.shares * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.direction == TradeDirection.LONG:
            return self.market_value - self.cost_basis
        else:  # SHORT
            return self.cost_basis - self.market_value
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L as percentage"""
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    def update_price(self, new_price: float):
        """Update current price"""
        self.current_price = new_price
    
    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss is None:
            return False
        
        if self.direction == TradeDirection.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit is None:
            return False
        
        if self.direction == TradeDirection.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit


@dataclass
class Portfolio:
    """
    Portfolio state snapshot
    
    Tracks equity, cash, positions, and portfolio metrics at a point in time
    """
    date: datetime
    cash: float
    positions: List[Position] = field(default_factory=list)
    
    @property
    def positions_value(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions)
    
    @property
    def equity(self) -> float:
        """Total portfolio equity (cash + positions value)"""
        return self.cash + self.positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions)
    
    @property
    def num_positions(self) -> int:
        """Number of open positions"""
        return len(self.positions)
    
    @property
    def leverage(self) -> float:
        """Portfolio leverage (positions_value / equity)"""
        if self.equity == 0:
            return 0.0
        return self.positions_value / self.equity
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker"""
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None
    
    def has_position(self, ticker: str) -> bool:
        """Check if portfolio has a position in ticker"""
        return self.get_position(ticker) is not None
    
    def to_dict(self) -> Dict:
        """Convert portfolio snapshot to dictionary"""
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'cash': self.cash,
            'positions_value': self.positions_value,
            'equity': self.equity,
            'unrealized_pnl': self.unrealized_pnl,
            'num_positions': self.num_positions,
            'leverage': self.leverage,
        }


@dataclass
class BacktestResult:
    """
    Complete backtest results container
    
    Stores all trades, portfolio snapshots, and metadata from a backtest run
    """
    # Backtest parameters
    ticker: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # Results
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    portfolio_snapshots: List[Portfolio] = field(default_factory=list)
    
    # Configuration used
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.1
    
    # Metadata
    strategy_name: str = "Default Strategy"
    strategy_params: Dict = field(default_factory=dict)
    
    @property
    def final_equity(self) -> float:
        """Final portfolio equity"""
        if len(self.equity_curve) == 0:
            return self.initial_capital
        return float(self.equity_curve.iloc[-1])
    
    @property
    def total_return(self) -> float:
        """Total return in dollars"""
        return self.final_equity - self.initial_capital
    
    @property
    def total_return_percent(self) -> float:
        """Total return as percentage"""
        return (self.total_return / self.initial_capital) * 100
    
    @property
    def num_trades(self) -> int:
        """Total number of trades"""
        return len(self.trades)
    
    @property
    def closed_trades(self) -> List[Trade]:
        """Get all closed trades"""
        return [t for t in self.trades if t.is_closed]
    
    @property
    def open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return [t for t in self.trades if t.is_open]
    
    @property
    def winning_trades(self) -> List[Trade]:
        """Get all winning trades"""
        return [t for t in self.closed_trades if t.pnl and t.pnl > 0]
    
    @property
    def losing_trades(self) -> List[Trade]:
        """Get all losing trades"""
        return [t for t in self.closed_trades if t.pnl and t.pnl < 0]
    
    @property
    def duration_days(self) -> int:
        """Backtest duration in days"""
        return (self.end_date - self.start_date).days
    
    def get_equity_at(self, date: datetime) -> float:
        """Get equity at specific date"""
        try:
            return float(self.equity_curve.loc[date])
        except KeyError:
            # Return last known equity before date
            mask = self.equity_curve.index <= date
            if mask.any():
                return float(self.equity_curve[mask].iloc[-1])
            return self.initial_capital
    
    def get_returns(self) -> pd.Series:
        """Calculate returns series"""
        if len(self.equity_curve) == 0:
            return pd.Series()
        return self.equity_curve.pct_change().fillna(0)
    
    def to_summary_dict(self) -> Dict:
        """Get summary statistics as dictionary"""
        return {
            'ticker': self.ticker,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'duration_days': self.duration_days,
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'total_return': self.total_return,
            'total_return_percent': self.total_return_percent,
            'num_trades': self.num_trades,
            'num_winning': len(self.winning_trades),
            'num_losing': len(self.losing_trades),
            'strategy_name': self.strategy_name,
        }

