from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
from techindicators import techindicators
import alpaca_trade_api
import pandas as pd
import numpy as np
import sys
import datetime
import base as b
import time
from scipy.stats.stats import pearsonr  
import matplotlib.pyplot as plt
draw=False
from alpha_vantage.timeseries import TimeSeries
from dateutil.parser import parse


def AddInfo(stock):
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
    percent_change = 100*(stock['close'][entry]-stock['open'][entry])/stock['open'][entry]
    percent_change30 = 100*(stock['adj_close'][entry]-day30['adj_close'][entry])/day30['adj_close'][entry]
    percent_change180 = 100*(stock['adj_close'][entry]-day180['adj_close'][entry])/day180['adj_close'][entry]
    percent_change365 = 100*(stock['adj_close'][entry]-day365['adj_close'][entry])/day365['adj_close'][entry]
    return [percent_change,percent_change30,percent_change180,percent_change365]

def formatInput(stock, ticker, rel_spy=[1.0,1.0,1.0,1.0], spy=None):
    # Add Information
    AddInfo(stock)
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
spy = runTicker(api,'SPY')
ticker='X'
ticker='TSLA'
stock_info = runTicker(api,ticker)
stock_info=runTickerAlpha(ts,ticker)
spy=runTickerAlpha(ts,'SPY')
print(spy['close'][0])
#spy['adj_close']/=spy['adj_close'][0]
print(spy)
#stock_info['adj_close']/=stock_info['adj_close'][0]
#stock_info['adj_close']/=spy['adj_close']
print(stock_info)

spy_info = GetPastPerformance(spy)
# build html table
columns = ['Ticker','% Change','% Change 30d','% Change 180d','% Change 1y','% Change 30d-SPY','% Change 1y-SPY','Corr. w/SPY','close','rsi10','CMF','sma10','sma20','sma100','sma200','rstd10']
entries=[]
entries+=[formatInput(spy, 'SPY',spy_info,spy=spy)]
j=0
for s in b.stock_lista:
    if s[0]=='SPY':
        continue
    if j%4==0 and j!=0:
        time.sleep(56)
    if j>50:
        break
    print(s[0])
    stock=runTickerAlpha(ts,s[0])
    entries+=[formatInput(stock, s[0],spy_info, spy=spy)]
    j+=1
#entries+=[formatInput(stock_info, ticker,spy_info,spy=spy)]

b.makeHTMLTable('stockinfo.html',columns=columns,entries=entries)

if draw:
    #plt.plot(stock_info.index,stock_info['close'])
    plt.plot(stock_info.index,stock_info['adj_close'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Closing price')
    plt.xlabel('Date')
    plt.show()
