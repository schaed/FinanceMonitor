#!/bin/bash
source /Users/schae/.bashrc
cd /Users/schae/testarea/finances/FinanceMonitor
source setup.sh
/opt/local/bin/python3.7 /Users/schae/testarea/finances/FinanceMonitor/TradingAlgo/News/bullish.py X >&/tmp/bullish_trading.log 
