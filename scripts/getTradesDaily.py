from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
import alpaca_trade_api
import pandas as pd
import numpy
import sys
import base as b

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg') 
from techindicators import techindicators

def GetTrades(ticker, ts, te, limit=50000):
    trades = None
    try:
        trades = api.get_trades(ticker, ts, te, limit=limit).df
        #trades = api.get_quotes(ticker, ts, te, limit=limit).df
    except alpaca_trade_api.rest.APIError:
        print('retry')
        trades = GetTrades(ticker, ts, te,limit).df
    if 'price' not in trades:
        print('error no entry for price')
        return []
        trades = GetTrades(ticker, ts, te,limit).df
        if 'price' not in trades:
            return []
    return trades

def GetQuotes(ticker, ts, te, limit=1000):
    quotes = None
    try:
        quotes = api.get_quotes(ticker, ts, te, limit=limit).df
    except alpaca_trade_api.rest.APIError:
        print('retry quotes')
        quotes = GetQuotes(ticker, ts, te, limit)
    if 'ask_price' not in quotes:
        #print(quotes)
        print('error no entry for ask price')
        return []
        quotes = GetQuotes(ticker, ts, te, limit)
        if 'ask_price' not in quotes:
            return []
    return quotes

# tf is time frame TimeFrame.Day, TimeFrame.Min, TimeFrame.Hour
# start time ts, end time te
def GetBars(ticker, tf, ts, te, limit=None):
    bars = None
    try:
        bars = api.get_bars(ticker, tf, ts, te, 'raw', limit=limit).df
    except alpaca_trade_api.rest.APIError:
        print('retry bars')
        print(bars)
        bars = GetBars(ticker, tf, ts, te, limit=limit)
    if 'open' not in bars:
        print(bars)
        print('error no entry for ask price')
        return []
        bars = GetBars(ticker, ts, te, limit=limit)
        if 'open' not in bars:
            return []
    return bars

def runTicker(api, ticker):
    trade_days = api.get_bars(ticker, TimeFrame.Day, "2021-01-02", "2021-03-16", 'raw').df
    trade_days = api.get_bars(ticker, TimeFrame.Day, "2021-03-16", "2021-03-16", 'raw').df
    print(ticker)
    print(trade_days)
    #print(pd.unique(trade_days['volume']))
    print(trade_days.index.unique())
    total_days_tested = len(trade_days.index.unique())
    days_less_thr=0
    days_less_thr5=0
    days_less_thr10=0
    days_less_thr15=0
    for myday in trade_days.index.unique():
        ts = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day,
                        hour = 16, minute=00, second = 0, tz = 'US/Eastern')
        te = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day,
                        hour = 20, minute=00, second = 0, tz = 'US/Eastern')
        #ts = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day,
        #                hour = 5, minute=00, second = 0, tz = 'US/Eastern')
        #te = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day,
        #                hour = 9, minute=00, second = 0, tz = 'US/Eastern')        
        #choseInd = [ind for ind in trade_days.index if (ind.day<9 or ind.hour>=16)]
        days_price = trade_days.loc[myday]
        print(days_price['high'])
        low_price=0.0
        try:
            low_price= float(days_price['high'].iloc[[0]])
            low_price= min(float(days_price['high'].iloc[[0]]),float(days_price['low'].iloc[[0]])) #,days_price['open'].iloc[[0]],days_price['close'].iloc[[0]])
        except:
            #print('broken input...skipping')
            low_price=days_price['high']
    
        print('prices today')
        print(ts.isoformat(),low_price)
        print(days_price)
        # collect the trades
        trades = GetTrades(ticker, ts.isoformat(), te.isoformat())
        if len(trades)<1:
            print('skipping day')
            continue
        min_price = float(trades['price'].min())
        print('min_price: %s' %min_price)
        if min_price<(0.96*low_price):
            days_less_thr+=1
            print('Low price on the day: %s' %(low_price))
            for i in range(0,min(50,len(trades))):
                print(trades.sort_values('price').iloc[[i]] )
                print(trades.sort_values('price').iloc[[i]]['conditions'] )
                print(trades.sort_values('price').iloc[[i]]['size'] )
            print(trades.sort_values('price').iloc[[-1]])
        if min_price<(0.95*low_price):
            days_less_thr5+=1    
        if min_price<(0.9*low_price):
            days_less_thr10+=1
        if min_price<(0.85*low_price):
            days_less_thr15+=1
        sys.stdout.flush()
    print('Summary %s:' %ticker)
    print('  %s tested days: %s' %(ticker,total_days_tested))
    print('  %s days below thr: %s' %(ticker,days_less_thr))
    print('  %s tested days: %s days below thr: %s, 5perc: %s, 10perc: %s, 15perc: %s' %(ticker, total_days_tested, days_less_thr,days_less_thr5,days_less_thr10,days_less_thr15))
    sys.stdout.flush()

api = REST('PKBYDGSHYI2IGLDVEGDY','1LpyIHmJXN7F7uT1C2ZPydjh9LcVbToBJD7E5e3Y')

ticker='TSLA'
#ticker='AAPL'
#ticker='RTX'
#ticker='JFU'
#ticker='MPC'
#ticker='GOOGL'
#ticker='F'
skip=True
if False:
    #for t in b.stock_lista:
    for t in b.stock_list:
    #for t in [[ticker]]:
    #if t[0]=='SCCO':
        if t[0]=='VIAB':
            skip=False
            continue
        if skip:
            continue
        if t[0].count('^'):
            continue
    for t in [[ticker]]:
        runTicker(api,t[0])

    sys.exit(0)

def GetMinuteInfo(ticker):
    bars = GetBars(ticker, TimeFrame.Minute, "2021-04-12", "2021-04-14",limit=None)
    print(bars)
    bars['vwap14']=techindicators.vwap(bars['high'],bars['low'],bars['close'],bars['volume'],14)
    bars['vwap10']=techindicators.vwap(bars['high'],bars['low'],bars['close'],bars['volume'],10)
    bars['vwap20']=techindicators.vwap(bars['high'],bars['low'],bars['close'],bars['volume'],60)
    # comparison to the market
    plt.clf()
    plt.plot(bars.index,bars['close'],color='black',label='Close')
    plt.plot(bars.index,bars['vwap10'],color='blue',label='vwap10')    
    plt.plot(bars.index,     bars['vwap14'],   color='red', label='vwap14')    
    plt.plot(bars.index,     bars['vwap20'],   color='green', label='vwap20')    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend(loc="upper left")
    plt.show()

#GetMinuteInfo(ticker)
    
month=4
day=14
start_date = "2021-0%s-%sT09:00:00-04:00" %(month,day)
end_date = "2021-0%s-%sT15:50:00-04:00" %(month,day)
start_date = "2021-0%s-%sT16:00:00-04:00" %(month,day)
end_date = "2021-0%s-%sT22:50:00-04:00" %(month,day)
trades = GetTrades(ticker, start_date, end_date,limit=500000) #api.get_trades("TSLA", start_date, end_date, limit=50000).df
quotes = GetQuotes(ticker, start_date, end_date,limit=5000) #api.get_trades("TSLA", start_date, end_date, limit=50000).df
print(quotes)
print(len(quotes))
#for i in range(0,min(50000,len(quotes))):
#    print(quotes.iloc[[i]] )

min_price = float(quotes['ask_price'].min())
max_size = float(quotes['ask_size'].max())
for i in range(0,min(50,len(quotes))):
    print(quotes.sort_values('ask_price').iloc[[i]] )
    print(quotes.sort_values('bid_price').iloc[[i]] )


print('')
bars = GetBars(ticker, TimeFrame.Day, "2021-01-02", "2021-04-14",limit=None)
print(bars)

print('')
low_price = 730.
print(trades)
min_price = float(trades['price'].min())
max_size = float(trades['size'].max())
print('min_price: %s' %min_price)
print('max_size: %s' %max_size)
if min_price<(0.96*low_price):
    print('Low price on the day: %s' %(low_price))
    for i in range(0,min(50,len(trades))):
        print(trades.sort_values('price').iloc[[i]] )
        print('conditions: %s' %((trades.sort_values('price')[['conditions','exchange']].iloc[[i]]) ))
        print('size: %s' %(trades.sort_values('price').iloc[[i]][['size']]) )
    print('Highest prices: %s' %((trades.sort_values('price')).iloc[[-1]][['price','exchange']]))
sys.exit(0)

trade_days = api.get_bars(ticker, TimeFrame.Day, "2021-01-02", "2021-03-12", 'raw').df
#trade_days = api.get_bars(ticker, TimeFrame.Day, "2021-03-05", "2021-03-12", 'raw').df
#print(trade_days)
#print(pd.unique(trade_days['volume']))
print(trade_days.index.unique())
total_days_tested = len(trade_days.index.unique())
days_less_thr=0
for myday in trade_days.index.unique():
    ts = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day, 
                    hour = 16, minute=00, second = 0, tz = 'US/Eastern') 
    te = pd.Timestamp(year = myday.year,  month = myday.month, day = myday.day, 
                    hour = 20, minute=00, second = 0, tz = 'US/Eastern') 
    #choseInd = [ind for ind in trade_days.index if (ind.day<9 or ind.hour>=16)]
    days_price = trade_days.loc[myday]
    print(days_price['high'])
    low_price=0.0
    try:
        low_price= float(days_price['high'].iloc[[0]])
        low_price= min(float(days_price['high'].iloc[[0]]),float(days_price['low'].iloc[[0]])) #,days_price['open'].iloc[[0]],days_price['close'].iloc[[0]])
    except:
        #print('broken input...skipping')
        low_price=days_price['high']

    print('prices today')    
    print(ts.isoformat(),low_price)
    print(days_price)
    # collect the trades
    trades = GetTrades(ticker, ts.isoformat(), te.isoformat())
    if len(trades)<1:
        print('skipping day')
        continue
    min_price = float(trades['price'].min())
    print('min_price: %s' %min_price)
    if min_price<(0.96*low_price):
        days_less_thr+=1
        print('Low price on the day: %s' %(low_price))
        for i in range(0,min(50,len(trades))):
            print(trades.sort_values('price').iloc[[i]] )
            print(trades.sort_values('price').iloc[[i]]['conditions'] )
            print(trades.sort_values('price').iloc[[i]]['size'] )
        print(trades.sort_values('price').iloc[[-1]])


print('Summary %s:' %ticker)
print('  tested days: %s' %(total_days_tested))

sys.exit(0)


print(trades)
#for j in range(0,len(trades)):
#    print(trades.iloc[[j]] )
print('min')
print(trades['price'].min())
for i in range(0,50):
            print(trades.sort_values('price').iloc[[i]] )
sys.exit(0)
#print(api.get_trades(ticker, "2021-03-11", "2021-03-11", limit=500).df)
for month in [1,2,3]:
    for day in range(1,30):
        start_date = "2021-0%s-%sT17:00:00-04:00" %(month,day)
        end_date = "2021-0%s-%sT22:50:00-04:00" %(month,day)
        trades = api.get_trades(ticker, start_date, end_date, limit=50000).df

        print('Date: %s' %(start_date))
        print(trades['price'].min())
        for i in range(0,50):
            print(trades.sort_values('price').iloc[[i]] )
        print(trades.sort_values('price').iloc[[-1]])
#print(api.get_quotes(ticker, "2021-03-11T16:30:00-04:00", "2021-03-11T20:50:00-04:00", limit=10).df)
#print(trades)
#print(trades[''])

#print(api.get_bars(ticker, TimeFrame.Hour, "2021-03-10", "2021-03-10", limit=10, adjustment='raw').df)
#print(api.get_bars("AAPL", TimeFrame.Hour, "2021-03-08", "2021-03-09", limit=10, adjustment='raw').df)
