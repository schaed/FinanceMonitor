# FinanceMonitor
FinanceMonitor


pip3 install alpaca_trade_api
pip3 install numpy
pip3 install pandas matplotlib mplfinance numpy_ext
pip3 install numpy numpy_ext pandas scipy TA-lib matplotlib alpha_vantage html5lib nltk cython zigzag --user


May need to install the vader libraries
import nltk
nltk.download('vader_lexicon')


For options pricing, try
https://www.barchart.com/stocks/quotes/X/options

# Runs to download earnings and stock news from the TheFly. Need to
# setup a daily cron job to download and run.
getNews.py

# downloads the earnings calendar and saves it as stockEarnings.csv
# Then it iterates this list and updates the daily stock prices into a
# database. Each stock is saved in its own table in stocksAV.db
# Also downloads the past earnings predictions and observations to
# earningsCalendar.db. both quarterlyEarnings and company overview
# (short info, etc with todays date) are saved
getEarnings.py

# Downloads data to stocksAV.db and writes an html table. 
buildTable.py

# Downloads data and plots many indicators. saves histograms and
# builds a webpage
channelTradingAll.py

# Building models and saving them
trainOnEarnings.py # build a NN to predict response. Mostly seems to
predict buy. It is bad at finding the tails
trainOnEarningsLogistic.py # multicategory NN training

# build database with earnings info. connects earnings and market
# indicators into a data base called earningsCalendarForTraining.db in
# a table call earningsInfo
analyzeEarnings.py

# scrap the web for short data
shortData.py


#
https://eresearch.fidelity.com/eresearch/conferenceCalls.jhtml?tab=earnings&begindate=4/29/2021
https://marketchameleon.com/Calendar/Earnings #
https://finance.yahoo.com/calendar/earnings/?day=2021-05-26


Way cheaper API
https://financialmodelingprep.com/developer/docs#Company-Quote

# has the time the data was delivered for the earnings
https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey=demo

# another free option. well free to start with low latency
https://iexcloud.io/docs/api/


# interesting to read
https://github.com/Syakyr/My-Trading-Project/tree/master/Risk%20Management
