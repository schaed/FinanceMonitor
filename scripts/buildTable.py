from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
from techindicators import techindicators # as techindicators
#import techindicators as techindicators
import alpaca_trade_api
import pandas as pd
import numpy as np
import sys,os
import datetime
import pickle
import base as b
import time
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
draw=False
from alpha_vantage.timeseries import TimeSeries
from dateutil.parser import parse
outdir = b.outdir
doStocks=True
loadFromPickle=False
def AddInfo(stock,market):
    stock['sma10']=techindicators.sma(stock['adj_close'],10)
    stock['sma20']=techindicators.sma(stock['adj_close'],20)
    if len(stock['adj_close'])>100:
        stock['sma100']=techindicators.sma(stock['adj_close'],100)
    else: stock['sma100']=np.zeros(len(stock['adj_close']))
    if len(stock['adj_close'])>200:    
        stock['sma200']=techindicators.sma(stock['adj_close'],200)
    else: stock['sma200']=np.zeros(len(stock['adj_close']))
    stock['rstd10']=techindicators.rstd(stock['adj_close'],10)
    stock['rsi10']=techindicators.rsi(stock['adj_close'],10)
    stock['cmf']=techindicators.cmf(stock['high'],stock['low'],stock['close'],stock['volume'],10)
    stock['KeltLower'],stock['KeltCenter'],stock['KeltUpper']=techindicators.kelt(stock['high'],stock['low'],stock['close'],20,2.0,20)
    stock['copp']=techindicators.copp(stock['close'],14,11,10)
    stock['daily_return']=stock['adj_close'].pct_change(periods=1)
    stock['daily_return_stddev14']=techindicators.rstd(stock['daily_return'],14)
    stock['beta']=techindicators.rollingBetav2(stock,14,market)
    stock['alpha']=techindicators.rollingAlpha(stock,14,market)
    #stock['rsquare']=techindicators.rollingRsquare(stock,14)
    stock['rsquare']=techindicators.rollingRsquare(stock,14,spy)
    stock['sharpe']=techindicators.sharpe(stock['daily_return'],30) # generally above 1 is good
    
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def GetTimeSlot(stock, days=180):
    today=datetime.datetime.now()
    past_date = today + datetime.timedelta(days=-1*days)
    date=stock.truncate(before=past_date)
    #date = stock[nearest(stock.index,past_date)]
    return date[:1]

def PercentageChange(percent_change):
    color='red'
    if percent_change>0.0: color='green'
    return b.colorHTML(percent_change,color)

def GetPastPerformance(stock):

    day180 = GetTimeSlot(stock)
    day30 = GetTimeSlot(stock,30)
    day365 = GetTimeSlot(stock,365)
    entry=-1
    input_list = ['sma10','sma20','sma100','sma200','rstd10']
    percent_change30=0.0; percent_change180=0.0;percent_change365=0.0
    percent_change = 100*(stock['close'][entry]-stock['open'][entry])/stock['open'][entry]
    if len(day30)>0:
        percent_change30 = 100*(stock['adj_close'][entry]-day30['adj_close'][entry])/day30['adj_close'][entry]
    if len(day180)>0:
        percent_change180 = 100*(stock['adj_close'][entry]-day180['adj_close'][entry])/day180['adj_close'][entry]
    if len(day365)>0:
        percent_change365 = 100*(stock['adj_close'][entry]-day365['adj_close'][entry])/day365['adj_close'][entry]
    return [percent_change,percent_change30,percent_change180,percent_change365]

def formatInput(stock, ticker, rel_spy=[1.0,1.0,1.0,1.0], spy=None):
    # Add Information
    try:
        AddInfo(stock,spy)
    except KeyError:
        print('ERROR processing %s' %ticker)
        return None
    past_perf = GetPastPerformance(stock)

    # compute the percentage changes
    percent_change180 = past_perf[2]
    percent_change30 = past_perf[1]
    percent_change365 = past_perf[3]
    percent_change = past_perf[0]
    percent_change30_rel_spy=(percent_change30-rel_spy[1])
    percent_change365_rel_spy=(percent_change365-rel_spy[3])

    info_list = [ticker]
    info_list += [PercentageChange(percent_change)]
    info_list += [PercentageChange(percent_change30)]
    info_list += [PercentageChange(percent_change180)]
    info_list += [PercentageChange(percent_change365)]
    info_list += [PercentageChange(percent_change30_rel_spy)]
    info_list += [PercentageChange(percent_change365_rel_spy)]
    min_length = min(len(spy),len(stock))
    info_list += [PercentageChange(pearsonr(spy['adj_close'][-min_length:],stock['adj_close'][-min_length:])[0])] # 0 is the correlation and 1 is the p-value
    
    # label the price as green if above the 200 day moving average
    entry=-1
    input_list = ['sma10','sma20','sma100','sma200','rstd10']
    color='red'
    if stock['close'][entry]>stock['sma200'][entry]:
        color='green'
    info_list += [b.colorHTML(stock['close'][entry],color)]            
    # label the price as green RSI
    color='black'
    if stock['rsi10'][entry]<20.0:
        color='red'
    if stock['rsi10'][entry]>80.0:
        color='green'
    info_list += [b.colorHTML(stock['rsi10'][entry],color,3)]
    # label the price as green CMF
    color='green'
    if stock['cmf'][entry]<0.0:
        color='red'
    info_list += [b.colorHTML(stock['cmf'][entry],color,3)]
    
    for j in input_list:
        info_list += [stock[j][entry]]
    for j in ['alpha','beta','sharpe','daily_return_stddev14','rsquare']:
        info_list += [b.colorHTML(stock[j][entry],'black',4)]
    return info_list
    
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def runTickerAlpha(ts, ticker):
    
    #a=ts.get_daily(ticker,'full')
    a=ts.get_daily_adjusted(ticker,'full')
    #print(a)
    a_new={}
    cols = ['Date','open','high','low','close','volume']
    cols = ['Date','open','high','low','close','adj_close','volume','dividendamt','splitcoef']
    my_floats = ['open','high','low','close']
    my_floats = ['open','high','low','close','adj_close','volume','dividendamt','splitcoef']
    
    #'5. adjusted close', '6. volume', '7. dividend amount', '8. split coefficient'
    for ki in cols:
        a_new[ki]=[]
    
    for entry in a:
        for key,i in entry.items():
            if not is_date(key):
                continue
            #print(key)
            a_new['Date']+=[key]
            ij=0
            todays_values = list(i.values())
            for j in ['open','high','low','close','adj_close','volume','dividendamt','splitcoef']:
                a_new[j]+=[todays_values[ij]]
                ij+=1
    # format
    output = pd.DataFrame(a_new)
    output['Date']=pd.to_datetime(output['Date'].astype(str), format='%Y-%m-%d')
    output['Date']=pd.to_datetime(output['Date'])
    output[my_floats]=output[my_floats].astype(float)
    output['volume'] = output['volume'].astype(np.int64)
    output = output.set_index('Date')
    output = output.sort_index()
    #print(output)
    return output
    
def runTicker(api, ticker):
    today=datetime.datetime.now()
    yesterday = today + datetime.timedelta(days=-1)
    d1 = yesterday.strftime("%Y-%m-%d")
    fouryao = (today + datetime.timedelta(days=-364*4.5)).strftime("%Y-%m-%d")  
    trade_days = api.get_bars(ticker, TimeFrame.Day, fouryao, d1, 'raw').df
    return trade_days
    #print(ticker)
    #print(trade_days)
ALPACA_ID = os.getenv('ALPACA_ID')
ALPACA_PAPER_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPHA_ID = os.getenv('ALPHA_ID')
api = REST(ALPACA_ID,ALPACA_PAPER_KEY)
ts = TimeSeries(key=ALPHA_ID)
#spy = runTicker(api,'SPY')
ticker='X'
ticker='TSLA'
#stock_info = runTicker(api,ticker)
#stock_info=runTickerAlpha(ts,ticker)
spy=None
if loadFromPickle and os.path.exists("SPY.p"):
    spy = pickle.load( open( "SPY.p", "rb" ) )    
else:
    spy=runTickerAlpha(ts,'SPY')
    pickle.dump( spy, open( "SPY.p", "wb" ) )
spy['daily_return']=spy['adj_close'].pct_change(periods=1)
print(spy['close'][0])
#spy['adj_close']/=spy['adj_close'][0]
print(spy)
#stock_info['adj_close']/=stock_info['adj_close'][0]
#stock_info['adj_close']/=spy['adj_close']
#print(stock_info)

spy_info = GetPastPerformance(spy)
# build html table
columns = ['Ticker','% Change','% Change 30d','% Change 180d','% Change 1y','% Change 30d-SPY','% Change 1y-SPY','Corr. w/SPY','close','rsi10','CMF','sma10','sma20','sma100','sma200','rstd10','alpha','beta','sharpe','daily_return_stddev14','rsquare']
entries=[]
entries+=[formatInput(spy, 'SPY',spy_info,spy=spy)]
j=0
#for s in b.stock_lista:
if doStocks:
    for s in b.stock_list:
        if s[0]=='SPY':
            continue
        if s[0].count('^'):
            continue
        if j%4==0 and j!=0:
            time.sleep(56)
        #if j>0:
        #    break
        print(s[0])
        sys.stdout.flush()    
        stock=None
        try:
            stock=runTickerAlpha(ts,s[0])
        except ValueError:
            j+=1
            continue
        stockInput = formatInput(stock, s[0],spy_info, spy=spy)
        if stockInput!=None:
            entries+=[stockInput]
        j+=1
    #entries+=[formatInput(stock_info, ticker,spy_info,spy=spy)]
    
    b.makeHTMLTable(outdir+'stockinfo.html',columns=columns,entries=entries)

# build the sector ETFs
columns=['Description']+columns
j=0
entries=[]
entries+=[['SPY']+formatInput(spy, 'SPY',spy_info,spy=spy)]
for s in b.etfs:
    if s[0]=='SPY':
        continue
    if j%3==0 and j!=0:
        time.sleep(56)
    #if j>1:
    #    break
    print(s[0])
    sys.stdout.flush()
    stock=None
    try:
        if loadFromPickle and os.path.exists("%s.p" %s[0]):
            stock = pickle.load( open( "%s.p" %s[0], "rb" ) )
        else:
            stock=runTickerAlpha(ts,s[0])
            pickle.dump( stock, open( "%s.p" %s[0], "wb" ) )
            j+=1
    except ValueError:
        print('ERROR processing...ValueError %s' %s[0])
        j+=1
        continue
    entries+=[[s[4]]+formatInput(stock, s[0],spy_info, spy=spy)]
b.makeHTMLTable(outdir+'sectorinfo.html',title='Sector Performance',columns=columns,entries=entries,linkIndex=1)

if draw:
    #plt.plot(stock_info.index,stock_info['close'])
    plt.plot(stock_info.index,stock_info['adj_close'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Closing price')
    plt.xlabel('Date')
    plt.show()
