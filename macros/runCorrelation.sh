#!/bin/bash
source /Users/schae/.bashrc
cd /Users/schae/testarea/finances/FinanceMonitor
source setup.sh
/opt/local/bin/python3.7 /Users/schae/testarea/finances/FinanceMonitor/scripts/relative_performance.py &> /tmp/run_corr.log
