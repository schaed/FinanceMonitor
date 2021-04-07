from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
import alpaca_trade_api
import pandas as pd
import numpy
import sys
import base as b

def GetTrades(ticker, ts, te, limit=50000):
    trades = None
    try:
        trades = api.get_trades(ticker, ts, te, limit=limit).df
    except alpaca_trade_api.rest.APIError:
        print('retry')
        trades = GetTrades(ticker, ts, te)
    if 'price' not in trades:
        print('error no entry for price')
        return []
        trades = GetTrades(ticker, ts, te)
        if 'price' not in trades:
            return []
    return trades

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

ALPACA_ID = os.getenv('ALPACA_ID')
ALPACA_PAPER_KEY = os.getenv('ALPACA_PAPER_KEY')
api = REST(ALPACA_ID,ALPACA_PAPER_KEY)


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

month=3
day=19
start_date = "2021-0%s-%sT09:00:00-04:00" %(month,day)
end_date = "2021-0%s-%sT15:50:00-04:00" %(month,day)
start_date = "2021-0%s-%sT16:00:00-04:00" %(month,day)
end_date = "2021-0%s-%sT20:50:00-04:00" %(month,day)
trades = GetTrades(ticker, start_date, end_date,limit=500000) #api.get_trades("TSLA", start_date, end_date, limit=50000).df
low_price = 666.
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
