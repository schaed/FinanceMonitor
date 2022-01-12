from techindicators import techindicators # as techindicators
#import techindicators as techindicators
from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker,SQL_CURSOR,ConfigTable,GetTimeSlot,IS_ALPHA_PREMIUM_WAIT_ITER,GetNNSelection,AddInfo
import pandas as pd
import numpy as np
import sys,os
import datetime
import pickle
import base as b
import time
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
draw=True
doPDFs=False
debug=False

def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None, axis = []):
    # plotting
    plt.clf()
    plt.scatter(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    if len(axis)>0:
        plt.axis(axis)
    plt.ylabel(yname)
    plt.xlabel(xname)
    if len(axis)>0:
        m, b = np.polyfit(xaxis.values, yaxis.values, 1) #m = slope, b=intercept
        plt.plot(xaxis.values, m*xaxis.values + b,color='red')
        print(m,b)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()


ticker='X'
j=0
readType='full'
sqlcursor = SQL_CURSOR()
ts = ALPHA_TIMESERIES()
api = ALPACA_REST()

stock_info,j=ConfigTable(ticker, sqlcursor,ts,readType, j)
trade_days = api.get_bars(ticker, TimeFrame.Minute, "2021-05-03", "2021-05-03", 'raw').df
#trade_days = api.get_bars(ticker, TimeFrame.Minute, "2021-04-30", "2021-05-03T12:17:00-04:00", 'raw').df    
trade_days = trade_days.tz_convert(tz='US/Eastern')


spy,j = ConfigTable('SPY', sqlcursor,ts,readType,hoursdelay=2)
AddInfo(spy,spy,debug=debug)
AddInfo(stock_info,spy,debug=debug)


stock_info['sma20d'] = stock_info['adj_close'] - stock_info['sma20']


stock_infoc = stock_info #GetTimeSlot(stock_info,days=70)
stock_infoc['daily_return'] = stock_infoc['adj_close'].pct_change()
stock_infoc['for_daily_return'] = stock_infoc['adj_close'].shift(-1).pct_change()
stock_infoc['openClose'] = stock_infoc['close'] - stock_infoc['open']
stock_infoc['openClose_next'] = stock_infoc['openClose'].shift(-1)

print(stock_infoc[['adj_close','daily_return','for_daily_return']])
plt.hist(stock_infoc['daily_return'],bins=50)
plt.yscale('log')
plt.show()
plt.hist(stock_infoc[stock_infoc['daily_return']>0.07]['for_daily_return'],bins=50)

plt.show()
plt.hist(stock_infoc[stock_infoc['daily_return']<-0.07]['for_daily_return'],bins=50)
plt.yscale('log')
plt.show()
print(stock_infoc[stock_infoc['daily_return']>0.07].describe())
print(stock_infoc[stock_infoc['daily_return']<-0.07].describe())
print(stock_infoc.describe())
stock_infoc = stock_infoc[stock_infoc['sma20d']>1.0]
MakePlot(stock_infoc['daily_return'], stock_infoc['for_daily_return'],
             xname='Todays return',yname='tomorrow',saveName='returnTest', hlines=[],title='returnTest',doSupport=False)
