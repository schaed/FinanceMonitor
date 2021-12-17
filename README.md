# FinanceMonitor
FinanceMonitor - various functionality from html tables to plotting market indicators to scraping webpages for news stories. Runs in python 3.7


```pip3 install alpaca_trade_api
pip3 install numpy
pip3 install pandas matplotlib mplfinance numpy_ext watchdog
pip3 install numpy numpy_ext pandas scipy TA-lib matplotlib alpha_vantage html5lib nltk cython zigzag lxml statsmodels pmdarima wget spacy nltk talib --user
python3 -m spacy download en_core_web_sm
```

Running machine learning requires some specific tensorflow+numpy libraries
```
python3.7 -m venv tflow-env
source tflow-env/bin/activate
export PYTHONPATH=/Users/schae/testarea/finances/FinanceMonitor/tflow-env/lib/python3.7/site-packages:$PYTHONPATH
pip3.7 install  "numpy==1.19.5" "tensorflow==2.5.0" # make sure to create an environment
```

May need to install the vader libraries
```
import nltk
nltk.download('vader_lexicon')
```

For options pricing, try
https://www.barchart.com/stocks/quotes/X/options

## Runs to download earnings and stock news from the TheFly. Need to setup a daily cron job to download and run.
```
getNews.py
```

### Downloads the earnings calendar and saves it as stockEarnings.csv. Then it iterates this list and updates the daily stock prices into a database. Each stock is saved in its own table in stocksAV.db Also downloads the past earnings predictions and observations to earningsCalendar.db. Both quarterlyEarnings and company overview (short info, etc with todays date) are saved
```
getEarnings.py
```

## Downloads data to stocksAV.db and writes an html table. 
```
buildTable.py
```
## Downloads data and plots many indicators. saves histograms and
## builds a webpage
```
channelTradingAll.py
```

## Building models and saving them.
```
trainOnEarnings.py # build a NN to predict response. Mostly seems to
predict buy. It is bad at finding the tails
trainOnEarningsLogistic.py # multicategory NN training
```

### Build database with earnings info. connects earnings and market indicators into a data base called earningsCalendarForTraining.db in a table call earningsInfo
```
analyzeEarnings.py
```

## scrap the web for short data
```
shortData.py
```

# Below are mostly notes on features to add
## Interesting websites to parse
```
https://eresearch.fidelity.com/eresearch/conferenceCalls.jhtml?tab=earnings&begindate=4/29/2021
https://marketchameleon.com/Calendar/Earnings #
https://finance.yahoo.com/calendar/earnings/?day=2021-05-26
```

## Way cheaper API
```
https://financialmodelingprep.com/developer/docs#Company-Quote
```

## Has the time the data was delivered for the earnings
```
https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey=demo
```

## Another free option. well free to start with low latency
```
https://iexcloud.io/docs/api/
```

## Interesting to read
```
https://github.com/Syakyr/My-Trading-Project/tree/master/Risk%20Management
```

## Add the data for company info and historical earnings analyze daily
   rates relative to the SMA. some kind of reversion to the mean. can
   I build a return probability  using the MA, bolanger bands, etc?
   Maybe do it on the 5m time scale add in fibs plotting with the
   zigzag

https://www.tensorflow.org/tutorials/structured_data/time_series
https://www.tensorflow.org/probability/examples/STS_approximate_inference_for_models_with_non_Gaussian_observations
https://www.tensorflow.org/probability/examples/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand

Interesting for commodity pricing
https://tradingeconomics.com/commodity/coffee

# finish implementing the trading of mean reversion
# downward fluctuation needs to have a volume spike, or trust this is
# a good stock or a good start to an upward trend
# plot the summary plots of the percentage of stocks above x average

Crontab
```
30 14 * * * /Users/schae/testarea/finances/googlefinance/getPrice.sh
58 14 * * * /Users/schae/testarea/finances/yahoo-finance/getPriceAll.sh
30  7 * * * /Users/schae/testarea/finances/FinanceMonitor/macros/getNews.sh
31  7 * * 1-6 /Users/schae/testarea/finances/FinanceMonitor/macros/getSummaryData.sh
32  15 * * 1-5 /Users/schae/testarea/finances/FinanceMonitor/macros/scanFinViz.sh
30  7 */10 * * /Users/schae/testarea/finances/FinanceMonitor/macros/getEarnings.sh
30  15 * * 1-5 /Users/schae/testarea/finances/FinanceMonitor/macros/runNewsTrading.sh
1  14 * * 1-5 /Users/schae/testarea/finances/FinanceMonitor/macros/runMeanReversion.sh
30 13 * * 2-6 lxplus /afs/cern.ch/user/s/schae/testarea/FinanceMonitor/run/runCorrelation.sh
```
