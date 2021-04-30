#!/bin/bash

cd /Users/schae/testarea/finances/FinanceMonitor
source setup.sh
python3.7 /Users/schae/testarea/finances/FinanceMonitor/scripts/getNews.py &> /tmp/news.log
