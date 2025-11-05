"""Signal output formatting"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from core.models import TradingSignal
from core.enums import SignalType


class SignalFormatter:
    """Format trading signals for display"""
    
    @staticmethod
    def print_signals(signals: List[TradingSignal], detailed: bool = False):
        """
        Print trading signals in a formatted way
        
        Args:
            signals: List of trading signals
            detailed: Whether to show detailed reasons
        """
        if not signals:
            print("\n[WARNING] No signals to display")
            return
        
        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"\n{'='*70}")
        print(f"TRADING SIGNAL REPORT — {timestamp}")
        print(f"{'='*70}\n")
        
        # Count signals by type
        signal_counts = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0
        }
        
        for signal in signals:
            signal_counts[signal.signal] = signal_counts.get(signal.signal, 0) + 1
        
        # Summary
        print(f"Total Analyzed: {len(signals)}")
        print(f"  BUY: {signal_counts[SignalType.BUY]}")
        print(f"  SELL: {signal_counts[SignalType.SELL]}")
        print(f"  HOLD: {signal_counts[SignalType.HOLD]}")
        print(f"\n{'-'*70}\n")
        
        # Individual signals
        for signal in signals:
            SignalFormatter._print_single_signal(signal, detailed)
    
    @staticmethod
    def _print_single_signal(signal: TradingSignal, detailed: bool):
        """Print a single trading signal"""
        # Signal header with color indicators
        signal_display = signal.signal.value
        if signal.signal == SignalType.BUY:
            indicator = "[BUY]"
        elif signal.signal == SignalType.SELL:
            indicator = "[SELL]"
        else:
            indicator = "[HOLD]"
        
        print(f"{signal.short_name} [{signal.ticker}]: {indicator}")
        print(f"  Signal: {signal.signal.value} (Confidence: {signal.confidence_percent:.1f}%)")
        print(f"  Last Close: ${signal.last_close:.2f} | ATR: ${signal.atr:.4f}")
        
        if signal.entry_zone:
            print(f"  Entry Zone: {signal.entry_zone}")
        
        if detailed and signal.reasons:
            print(f"  Analysis:")
            for reason in signal.reasons:
                print(f"    - {reason}")
        
        print()  # Blank line between signals
    
    @staticmethod
    def print_actionable_only(signals: List[TradingSignal], detailed: bool = False):
        """Print only actionable signals (BUY/SELL)"""
        actionable = [s for s in signals if s.is_actionable]
        
        if not actionable:
            print("\n[INFO] No actionable signals at this time")
            return
        
        print(f"\n{'='*70}")
        print(f"ACTIONABLE SIGNALS ONLY ({len(actionable)} signals)")
        print(f"{'='*70}\n")
        
        for signal in actionable:
            SignalFormatter._print_single_signal(signal, detailed)
    
    @staticmethod
    def export_to_csv(signals: List[TradingSignal], filepath: str):
        """
        Export signals to CSV file
        
        Args:
            signals: List of trading signals
            filepath: Output file path (automatically saved to results folder with date)
        """
        import csv
        
        # Create results folder if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Add date prefix to filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = Path(filepath).name  # Get just the filename
        dated_filepath = results_dir / f"{date_str}_{filename}"
        
        with open(dated_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Ticker', 'Name', 'Signal', 'Confidence %',
                'Last Close', 'ATR', 'Entry Low', 'Entry High',
                'Timeframe'
            ])
            
            # Data rows
            for signal in signals:
                writer.writerow([
                    signal.ticker,
                    signal.short_name,
                    signal.signal.value,
                    f"{signal.confidence_percent:.2f}",
                    f"{signal.last_close:.2f}",
                    f"{signal.atr:.4f}",
                    f"{signal.entry_zone.lower_bound:.2f}" if signal.entry_zone else "",
                    f"{signal.entry_zone.upper_bound:.2f}" if signal.entry_zone else "",
                    signal.timeframe.value
                ])
        
        print(f"\n[SUCCESS] Signals exported to: {dated_filepath}")
    
    @staticmethod
    def export_combined_to_csv(us_signals: List[TradingSignal], 
                                sg_signals: List[TradingSignal]):
        """
        Export combined US and SG signals to a single CSV file
        
        Args:
            us_signals: List of US trading signals
            sg_signals: List of SG trading signals
        """
        import csv
        
        # Create results folder if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create combined filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        dated_filepath = results_dir / f"{date_str}_combined_signals.csv"
        
        all_signals = us_signals + sg_signals
        
        with open(dated_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Market', 'Ticker', 'Name', 'Signal', 'Confidence %',
                'Last Close', 'ATR', 'Entry Low', 'Entry High',
                'Timeframe'
            ])
            
            # US signals
            for signal in us_signals:
                writer.writerow([
                    'US',
                    signal.ticker,
                    signal.short_name,
                    signal.signal.value,
                    f"{signal.confidence_percent:.2f}",
                    f"{signal.last_close:.2f}",
                    f"{signal.atr:.4f}",
                    f"{signal.entry_zone.lower_bound:.2f}" if signal.entry_zone else "",
                    f"{signal.entry_zone.upper_bound:.2f}" if signal.entry_zone else "",
                    signal.timeframe.value
                ])
            
            # SG signals
            for signal in sg_signals:
                writer.writerow([
                    'SG',
                    signal.ticker,
                    signal.short_name,
                    signal.signal.value,
                    f"{signal.confidence_percent:.2f}",
                    f"{signal.last_close:.2f}",
                    f"{signal.atr:.4f}",
                    f"{signal.entry_zone.lower_bound:.2f}" if signal.entry_zone else "",
                    f"{signal.entry_zone.upper_bound:.2f}" if signal.entry_zone else "",
                    signal.timeframe.value
                ])
        
        print(f"[SUCCESS] Combined signals exported to: {dated_filepath}")
    
    @staticmethod
    def export_to_json(signals: List[TradingSignal], filepath: str):
        """
        Export signals to JSON file
        
        Args:
            signals: List of trading signals
            filepath: Output file path (automatically saved to results folder with date)
        """
        # Create results folder if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Add date prefix to filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = Path(filepath).stem  # Get filename without extension
        dated_filepath = results_dir / f"{date_str}_{filename}.json"
        
        # Convert signals to dictionary format
        data = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_analyzed": len(signals),
            "summary": {
                "buy": sum(1 for s in signals if s.signal == SignalType.BUY),
                "sell": sum(1 for s in signals if s.signal == SignalType.SELL),
                "hold": sum(1 for s in signals if s.signal == SignalType.HOLD)
            },
            "signals": []
        }
        
        for signal in signals:
            signal_dict = {
                "ticker": signal.ticker,
                "name": signal.short_name,
                "signal": signal.signal.value,
                "confidence": round(signal.confidence_percent, 2),
                "last_close": round(signal.last_close, 2),
                "atr": round(signal.atr, 4),
                "entry_zone": {
                    "lower": round(signal.entry_zone.lower_bound, 2) if signal.entry_zone else None,
                    "upper": round(signal.entry_zone.upper_bound, 2) if signal.entry_zone else None
                } if signal.entry_zone else None,
                "timeframe": signal.timeframe.value,
                "is_actionable": signal.is_actionable,
                "analysis": signal.reasons if signal.reasons else []
            }
            data["signals"].append(signal_dict)
        
        # Write to JSON file with pretty formatting
        with open(dated_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Signals exported to JSON: {dated_filepath}")
    
    @staticmethod
    def export_combined_to_json(us_signals: List[TradingSignal], 
                                 sg_signals: List[TradingSignal]):
        """
        Export combined US and SG signals to a single JSON file
        
        Args:
            us_signals: List of US trading signals
            sg_signals: List of SG trading signals
        """
        # Create results folder if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create combined filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        dated_filepath = results_dir / f"{date_str}_combined_signals.json"
        
        # Convert signals to dictionary format
        data = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_analyzed": len(us_signals) + len(sg_signals),
            "markets": {
                "us": {
                    "total": len(us_signals),
                    "buy": sum(1 for s in us_signals if s.signal == SignalType.BUY),
                    "sell": sum(1 for s in us_signals if s.signal == SignalType.SELL),
                    "hold": sum(1 for s in us_signals if s.signal == SignalType.HOLD)
                },
                "sg": {
                    "total": len(sg_signals),
                    "buy": sum(1 for s in sg_signals if s.signal == SignalType.BUY),
                    "sell": sum(1 for s in sg_signals if s.signal == SignalType.SELL),
                    "hold": sum(1 for s in sg_signals if s.signal == SignalType.HOLD)
                }
            },
            "summary": {
                "total_buy": sum(1 for s in (us_signals + sg_signals) if s.signal == SignalType.BUY),
                "total_sell": sum(1 for s in (us_signals + sg_signals) if s.signal == SignalType.SELL),
                "total_hold": sum(1 for s in (us_signals + sg_signals) if s.signal == SignalType.HOLD)
            },
            "us_signals": [],
            "sg_signals": []
        }
        
        # Add US signals
        for signal in us_signals:
            signal_dict = {
                "ticker": signal.ticker,
                "name": signal.short_name,
                "signal": signal.signal.value,
                "confidence": round(signal.confidence_percent, 2),
                "last_close": round(signal.last_close, 2),
                "atr": round(signal.atr, 4),
                "entry_zone": {
                    "lower": round(signal.entry_zone.lower_bound, 2) if signal.entry_zone else None,
                    "upper": round(signal.entry_zone.upper_bound, 2) if signal.entry_zone else None
                } if signal.entry_zone else None,
                "timeframe": signal.timeframe.value,
                "is_actionable": signal.is_actionable,
                "analysis": signal.reasons if signal.reasons else []
            }
            data["us_signals"].append(signal_dict)
        
        # Add SG signals
        for signal in sg_signals:
            signal_dict = {
                "ticker": signal.ticker,
                "name": signal.short_name,
                "signal": signal.signal.value,
                "confidence": round(signal.confidence_percent, 2),
                "last_close": round(signal.last_close, 2),
                "atr": round(signal.atr, 4),
                "entry_zone": {
                    "lower": round(signal.entry_zone.lower_bound, 2) if signal.entry_zone else None,
                    "upper": round(signal.entry_zone.upper_bound, 2) if signal.entry_zone else None
                } if signal.entry_zone else None,
                "timeframe": signal.timeframe.value,
                "is_actionable": signal.is_actionable,
                "analysis": signal.reasons if signal.reasons else []
            }
            data["sg_signals"].append(signal_dict)
        
        # Write to JSON file with pretty formatting
        with open(dated_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Combined signals exported to JSON: {dated_filepath}")

