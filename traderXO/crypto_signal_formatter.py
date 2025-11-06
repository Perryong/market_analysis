"""
Crypto Signal Formatter
Display cryptocurrency signals in the same format as stock signals
"""

from typing import List
from traderXO.crypto_signal_analyzer import CryptoSignal


class CryptoSignalFormatter:
    """Format crypto trading signals for display"""
    
    @staticmethod
    def print_signal(signal: CryptoSignal, detailed: bool = True):
        """
        Print a single crypto signal in stock analysis format
        
        Args:
            signal: CryptoSignal to display
            detailed: Show detailed analysis
        """
        # Header with signal
        print(f"\n{signal.ticker} [{signal.ticker}]: [{signal.signal}]")
        
        # Confidence and price info
        print(f"  Signal: {signal.signal} (Confidence: {signal.confidence:.1f}%)")
        print(f"  Last Close: ${signal.last_close:.2f} | ATR: ${signal.atr:.4f}")
        print(f"  Entry Zone: ${signal.entry_zone_low:.2f} - ${signal.entry_zone_high:.2f}")
        
        # Detailed analysis
        if detailed and signal.analysis_details:
            print(f"  Analysis:")
            for detail in signal.analysis_details:
                print(f"    - {detail}")
    
    @staticmethod
    def print_signals(signals: List[CryptoSignal], detailed: bool = True):
        """
        Print multiple crypto signals
        
        Args:
            signals: List of CryptoSignals
            detailed: Show detailed analysis
        """
        if not signals:
            print("\n[WARNING] No crypto signals to display")
            return
        
        # Header
        print(f"\n{'='*70}")
        print(f"SIGNAL REPORT")
        print(f"{'='*70}\n")
        
        # Count signals by type
        buy_count = sum(1 for s in signals if s.signal == 'BUY')
        sell_count = sum(1 for s in signals if s.signal == 'SELL')
        hold_count = sum(1 for s in signals if s.signal == 'HOLD')
        
        # Summary
        print(f"Total Analyzed: {len(signals)}")
        print(f"  BUY: {buy_count}")
        print(f"  SELL: {sell_count}")
        print(f"  HOLD: {hold_count}")
        print(f"\n{'-'*70}")
        
        # Individual signals
        for signal in signals:
            CryptoSignalFormatter.print_signal(signal, detailed)
        
        print(f"\n{'='*70}\n")
    
    @staticmethod
    def print_actionable_only(signals: List[CryptoSignal], detailed: bool = False):
        """
        Print only actionable signals (BUY/SELL)
        
        Args:
            signals: List of CryptoSignals
            detailed: Show detailed analysis
        """
        actionable = [s for s in signals if s.signal in ['BUY', 'SELL']]
        
        if not actionable:
            print("\n[INFO] No actionable crypto signals (all HOLD)")
            return
        
        print(f"\n{'='*70}")
        print(f"ACTIONABLE CRYPTO SIGNALS")
        print(f"{'='*70}")
        
        for signal in actionable:
            CryptoSignalFormatter.print_signal(signal, detailed)
    
    @staticmethod
    def get_signal_summary(signals: List[CryptoSignal]) -> dict:
        """
        Get summary statistics for crypto signals
        
        Args:
            signals: List of CryptoSignals
            
        Returns:
            Dictionary with summary stats
        """
        if not signals:
            return {
                'total': 0,
                'buy': 0,
                'sell': 0,
                'hold': 0,
                'avg_confidence': 0,
                'actionable': 0
            }
        
        return {
            'total': len(signals),
            'buy': sum(1 for s in signals if s.signal == 'BUY'),
            'sell': sum(1 for s in signals if s.signal == 'SELL'),
            'hold': sum(1 for s in signals if s.signal == 'HOLD'),
            'avg_confidence': sum(s.confidence for s in signals) / len(signals),
            'actionable': sum(1 for s in signals if s.signal in ['BUY', 'SELL'])
        }
    
    @staticmethod
    def export_to_csv(signals: List[CryptoSignal], filename: str = 'crypto_signals.csv'):
        """
        Export crypto signals to CSV
        
        Args:
            signals: List of CryptoSignals
            filename: Output filename
        """
        import csv
        from pathlib import Path
        from datetime import datetime
        
        # Create output directory
        output_dir = Path('traderXO/results')
        output_dir.mkdir(exist_ok=True)
        
        # Add date prefix
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = output_dir / f"{date_str}_{filename}"
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Ticker', 'Signal', 'Confidence %',
                'Last Close', 'ATR', 'Entry Low', 'Entry High',
                'RSI', 'MACD', 'Delta', 'Cumulative Delta',
                'EMA 12', 'EMA 21', 'Volume'
            ])
            
            # Data rows
            for signal in signals:
                ms = signal.market_structure
                writer.writerow([
                    signal.ticker,
                    signal.signal,
                    f"{signal.confidence:.2f}",
                    f"{signal.last_close:.2f}",
                    f"{signal.atr:.4f}",
                    f"{signal.entry_zone_low:.2f}",
                    f"{signal.entry_zone_high:.2f}",
                    f"{ms.get('rsi', 0):.2f}",
                    f"{ms.get('macd', 0):.4f}",
                    f"{ms.get('delta', 0):.0f}",
                    f"{ms.get('cumulative_delta', 0):.0f}",
                    f"{ms.get('ema_12', 0):.2f}",
                    f"{ms.get('ema_21', 0):.2f}",
                    f"{ms.get('volume', 0):.0f}"
                ])
        
        print(f"\n[SUCCESS] Crypto signals exported to: {filepath}")
        return filepath

