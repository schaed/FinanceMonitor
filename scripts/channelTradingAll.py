from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
from techindicators import techindicators
#import techindicators as techindicators
import alpaca_trade_api
import pandas as pd
import numpy as np
import sys
import datetime
import base as b
import pickle
import time
import os
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import mplfinance as mpf
draw=False
from alpha_vantage.timeseries import TimeSeries
from dateutil.parser import parse
outdir = b.outdir
def CandleStick(data, ticker):

    # Extracting Data for plotting
    #data = pd.read_csv('candlestick_python_data.csv')
    df = data.loc[:, ['open', 'high', 'low', 'close','volume']]
    df.columns = ['Open', 'High', 'Low', 'Close','Volume']
    df['UpperB'] = data['BolUpper']        
    df['LowerB'] = data['BolLower']
    df['KeltLower'] = data['KeltLower']        
    df['KeltUpper'] = data['KeltUpper']
    df['sma200'] = data['sma200']

    # Plot candlestick.
    # Add volume.
    # Add moving averages: 3,6,9.
    # Save graph to *.png.
    ap0 = [ mpf.make_addplot(df['UpperB'],color='g'),  # uses panel 0 by default
        mpf.make_addplot(df['LowerB'],color='b'),  # uses panel 0 by default
        mpf.make_addplot(df['sma200'],color='r'),  # uses panel 0 by default        
        mpf.make_addplot(df['KeltLower'],color='darkviolet'),  # uses panel 0 by default
        mpf.make_addplot(df['KeltUpper'],color='magenta'),  # uses panel 0 by default
      ]
    #mpf.plot(df,type='candle',volume=True,addplot=ap0) 
    fig,axes=mpf.plot(df, type='candle', style='charles',
            title=ticker,
            ylabel='Price ($) %s' %ticker,
            ylabel_lower='Shares \nTraded',
            volume=True, 
            mav=(200),
            addplot=ap0,
            returnfig=True,
            savefig=outdir+'test-mplfiance_'+ticker+'.pdf')
        # Configure chart legend and title
    axes[0].legend(['Price','Bolanger Up','Bolanger Down','SMA200','Kelt+','Kelt-'])
    #axes[0].set_title(ticker)
    # Save figure to file
    fig.savefig(outdir+'test-mplfiance_'+ticker+'.pdf')
    fig.savefig(outdir+'test-mplfiance_'+ticker+'.png')
    techindicators.plot_support_levels(ticker,df,[mpf.make_addplot(df['sma200'],color='r') ],outdir=outdir)
    # adds below as a sub-plot
    #ap2 = [ mpf.make_addplot(df['UpperB'],color='g',panel=2),  # panel 2 specified
    #        mpf.make_addplot(df['LowerB'],color='b',panel=2),  # panel 2 specified
    #    ]
    #mpf.plot(df,type='candle',volume=True,addplot=ap2)
    #plt.savefig('CandleStick.pdf')
    #mpf.plot(df,tlines=[dict(tlines=datepairs,tline_use='high',colors='g'),
    #                dict(tlines=datepairs,tline_use='low',colors='b'),
    #                dict(tlines=datepairs,tline_use=['open','close'],colors='r')],
    #     figscale=1.35
    #    )
    
def GetTimeSlot(stock, days=365):
    today=datetime.datetime.now()
    past_date = today + datetime.timedelta(days=-1*days)
    date=stock.truncate(before=past_date)
    #date = stock[nearest(stock.index,past_date)]
    return date
def DrawPlots(stock_info,ticker):
    #plt.plot(stock_info.index,stock_info['close'])
    techindicators.supportLevels(stock_info)
    if not draw:
        plt.ioff()
    plt.plot(stock_info.index,stock_info['adj_close'])
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Closing price')
    plt.xlabel('Date')
    if draw: plt.show()
    plt.savefig(outdir+'price_support_%s.pdf' %ticker)
    plt.savefig(outdir+'price_support_%s.png' %ticker)
    if not draw: plt.close()
    plt.plot(stock_info.index,stock_info['copp'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Coppuck Curve')
    plt.xlabel('Date')
    plt.hlines(0.0,xmin=min(stock_info.index), xmax=max(stock_info.index),colors='black')
    if draw: plt.show()
    plt.savefig(outdir+'copp_%s.pdf' %ticker)
    plt.savefig(outdir+'copp_%s.png' %ticker)
    if not draw: plt.close()
    plt.plot(stock_info.index,stock_info['sharpe'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Date')
    plt.hlines(0.0,xmin=min(stock_info.index), xmax=max(stock_info.index),colors='black')
    if draw: plt.show()
    plt.savefig(outdir+'sharpe_%s.pdf' %ticker)
    plt.savefig(outdir+'sharpe_%s.png' %ticker)
    if not draw: plt.close()
    plt.plot(stock_info.index,stock_info['beta'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Beta')
    plt.xlabel('Date')
    if draw: plt.show()
    plt.savefig(outdir+'beta_%s.pdf' %ticker)
    plt.savefig(outdir+'beta_%s.png' %ticker)
    if not draw: plt.close()
    plt.plot(stock_info.index,stock_info['alpha'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Alpha')
    plt.xlabel('Date')
    plt.hlines(0.0,xmin=min(stock_info.index), xmax=max(stock_info.index),colors='black')
    plt.title(' Alpha')
    if draw: plt.show()
    plt.savefig(outdir+'alpha_%s.pdf' %ticker)
    plt.savefig(outdir+'alpha_%s.png' %ticker)
    if not draw: plt.close()
    plt.plot(stock_info.index,stock_info['rsquare'])    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('R-squared')
    plt.xlabel('Date')
    plt.hlines(0.7,xmin=min(stock_info.index), xmax=max(stock_info.index),colors='black')
    if draw: plt.show()
    plt.savefig(outdir+'rsquare_%s.pdf' %ticker)    
    plt.savefig(outdir+'rsquare_%s.png' %ticker)
    if not draw: plt.close()
    CandleStick(stock_info,ticker)
    
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
    stock['BolLower'],stock['BolCenter'],stock['BolUpper']=techindicators.boll(stock['adj_close'],20,2.0,5)
    stock['KeltLower'],stock['KeltCenter'],stock['KeltUpper']=techindicators.kelt(stock['high'],stock['low'],stock['close'],20,2.0,20)
    stock['copp']=techindicators.copp(stock['close'],14,11,10)
    stock['daily_return']=stock['adj_close'].pct_change(periods=1)
    stock['daily_return_stddev14']=techindicators.rstd(stock['daily_return'],14)
    stock['beta']=techindicators.rollingBetav2(stock,14,market)
    stock['alpha']=techindicators.rollingAlpha(stock,14,market)        
    stock['rsquare']=techindicators.rollingRsquare(stock,14,spy)

    stock['sharpe']=techindicators.sharpe(stock['daily_return'],30) # generally above 1 is good
    
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
ticker='TSLA'
#ticker='TSLA'
stock_info=None
spy=None
#if os.path.exists("%s.p" %ticker):
#    stock_info = pickle.load( open( "%s.p" %ticker, "rb" ) )
#    spy = pickle.load( open( "SPY.p", "rb" ) )
#else:
#stock_info = runTicker(api,ticker)
stock_info=runTickerAlpha(ts,ticker)
spy=runTickerAlpha(ts,'SPY')
#pickle.dump( spy, open( "SPY.p", "wb" ) )
#pickle.dump( stock_info, open( "%s.p" %ticker, "wb" ) )
# add info
if len(stock_info)==0:
    print('ERROR - empy info %s' %ticker)
spy['daily_return']=spy['adj_close'].pct_change(periods=1)

j=0
cdir = os.getcwd()
for s in b.stock_list:
    if s[0]=='SPY':
        continue
    if s[0].count('^'):
        continue
    if j%4==0 and j!=0:
        time.sleep(56)
    print(s[0])
    sys.stdout.flush()
    stock_info=None
    #if j>2:
    #    break
    try:
        stock_info=runTickerAlpha(ts,s[0])
    except ValueError:
        j+=1
        continue
    AddInfo(stock_info, spy)
    stock_info = GetTimeSlot(stock_info) # gets the one year timeframe
    DrawPlots(stock_info,s[0])
    os.chdir(outdir)
    b.makeHTML('%s.html' %s[0],s[0],filterPattern='*_%s' %s[0],describe=s[4])
    os.chdir(cdir)    
    j+=1

