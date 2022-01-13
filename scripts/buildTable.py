from techindicators import techindicators # as techindicators
#import techindicators as techindicators
from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker,SQL_CURSOR,ConfigTable,GetTimeSlot,IS_ALPHA_PREMIUM_WAIT_ITER,GetNNSelection
import pandas as pd
import numpy as np
import sys,os
import datetime
import pickle
import base as b
import time
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
debug=False
draw=False
outdir = b.outdir
doStocks=True
loadFromPickle=False
loadSQL = True
readType='full'
import zigzag
from zigzag import *

sqlcursorShort = SQL_CURSOR(db_name='stocksShort.db')
sqlcursorExtra = SQL_CURSOR(db_name='stocksShortExtra.db')
def readShortInfo(ticker):
    stock=None
    try:
        stock = pd.read_sql('SELECT * FROM %s' %ticker, sqlcursorExtra) #,index_col='Date')
        stock['LogDate']=pd.to_datetime(stock.LogDate.astype(str), format='%Y-%m-%d')
        stock['LogDate']=pd.to_datetime(stock['LogDate'])
        stock = stock.set_index('LogDate')
        stock = stock.sort_index()
        entryShort=0
        if len(stock)>0:
            return stock.iloc[-1][['Insider Own','Inst Own','Short Float','Rel Volume']].values
    except pd.io.sql.DatabaseError:
        #return [0,0,0,0]
        return ['N/A','N/A','N/A','N/A']
    return ['N/A','N/A','N/A','N/A']

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
    stock['cci']=techindicators.cci(stock['high'],stock['low'],stock['close'],20) 
    stock['stochK'],stock['stochD']=techindicators.stoch(stock['high'],stock['low'],stock['close'],14,3,3)    
    stock['obv']=techindicators.obv(stock['adj_close'],stock['volume'])
    stock['force']=techindicators.force(stock['adj_close'],stock['volume'],13)
    stock['macd'],stock['macdsignal']=techindicators.macd(stock['adj_close'],12,26,9)
    stock['market'] = market['adj_close']
    stock['corr14']=stock['adj_close'].rolling(14).corr(spy['market'])
    #stock['pdmd'],stock['ndmd'],stock['adx']=techindicators.adx(stock['high'],stock['low'],stock['close'],14)
    #stock['aroonUp'],stock['aroonDown'],stock['aroon']=techindicators.aroon(stock['high'],stock['low'],25)
    stock['vwap14']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],14)
    stock['vwap10']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],10)
    stock['vwap20']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],20)
    stock['chosc']=techindicators.chosc(stock['high'],stock['low'],stock['close'],stock['volume'],3,10)
    stock['vwap10diff'] = (stock['adj_close'] - stock['vwap10'])/stock['adj_close']
    #stock['max_drawdown'] = stock['adj_close'].rolling(250).apply(max_drawdown)
    day365 = GetTimeSlot(stock,365)    
    stock['max_drawdown'] = np.ones(len(stocks))*zigzag.max_drawdown(day365['adj_close'].values)
    #print(stock.max_drawdown)
    
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def PercentageChange(percent_change):
    color='red'
    if percent_change>0.0: color='green'
    return b.colorHTML(percent_change,color)

def GetPastPerformance(stock):

    day180 = GetTimeSlot(stock)
    day30 = GetTimeSlot(stock,30)
    day365 = GetTimeSlot(stock,365)
    entry=-1
    entry_old=0
    input_list = ['sma10','sma20','sma100','sma200','rstd10']
    percent_change30=0.0; percent_change180=0.0;percent_change365=0.0
    percent_change = 100*(stock['close'][entry]-stock['open'][entry])/stock['open'][entry]
    if len(day30)>0:
        percent_change30 = 100*(stock['adj_close'][entry]-day30['adj_close'][entry_old])/day30['adj_close'][entry]
    if len(day180)>0:
        percent_change180 = 100*(stock['adj_close'][entry]-day180['adj_close'][entry_old])/day180['adj_close'][entry]
    if len(day365)>0:
        percent_change365 = 100*(stock['adj_close'][entry]-day365['adj_close'][entry_old])/day365['adj_close'][entry]
    #print([percent_change,percent_change30,percent_change180,percent_change365])
    return [percent_change,percent_change30,percent_change180,percent_change365]

def formatInput(stock, ticker, rel_spy=[1.0,1.0,1.0,1.0], spy=None):
    # Add Information
    if 'cmf' not in stock.columns:
        try:
            AddInfo(stock,spy)
        except (KeyError,ValueError):
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
    input_list = ['sma10','sma20','sma100','sma200','rstd10','cci','chosc','force','pred']
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
    for j in ['alpha','beta','sharpe','daily_return_stddev14','rsquare','vwap10diff','corr14','max_drawdown']:
        if j not in stock:
            print('Error could not find %s' %j)
            #print(stock.columns)
            print(ticker)
            stock[j] = stock['alpha']
        info_list += [b.colorHTML(stock[j][entry],'black',4)]
    info_list+=list(readShortInfo(ticker))
    return info_list
    
api = ALPACA_REST()
ts = ALPHA_TIMESERIES()
ticker='X'
ticker='TSLA'
#stock_info = runTicker(api,ticker)
#stock_info=runTickerAlpha(ts,ticker)
sqlcursor = SQL_CURSOR()
connectionCal = SQL_CURSOR('earningsCalendar.db')
#spyB=None
#if loadFromPickle and os.path.exists("SPY.p"):
#    spyB = pickle.load( open( "SPY.p", "rb" ) )
#else:
#    spyB=runTickerAlpha(ts,'SPY')
#    pickle.dump( spyB, open( "SPY.p", "wb" ) )
#spyB['daily_return']=spyB['adj_close'].pct_change(periods=1)
j=0
spy,j=GetNNSelection('SPY', ts, connectionCal, sqlcursor, None,
                       debug=debug, j=j,addRangeOpens=False,
                       training_dir='models/',training_name='stockEarningsModelTestv2noEPS')

print(spy['close'][0])
print(spy)
n_ALPHA_PREMIUM_WAIT_ITER = IS_ALPHA_PREMIUM_WAIT_ITER()
spy_info = GetPastPerformance(spy)
# build html table
columns = ['Ticker','% Change','% Change 30d','% Change 180d','% Change 1y','% Change 30d-SPY','% Change 1y-SPY','Corr. w/SPY','close','rsi10','CMF','sma10','sma20','sma100','sma200','rstd10','CCI','ChaikinOsc','Force Idx','Pred','alpha','beta','sharpe','daily_return_stddev14','rsquare','vwap10','SPY Corr 14d','Max DrawDown','Insider Own','Inst Own','Short Float','Rel Volume']
entries=[]
entries+=[formatInput(spy, 'SPY',spy_info,spy=spy)]
j=0
#for s in b.stock_lista:
if doStocks:
    for s in b.stock_list:
        if s[0]=='SPY':
            continue
        readShortInfo(s[0])
        if s[0].count('^'):
            continue
        if j%n_ALPHA_PREMIUM_WAIT_ITER==0 and j!=0:
            time.sleep(56)
        #if j>0:
        #    break
        print(s[0])
        sys.stdout.flush()    
        stock=None
        stock,j=GetNNSelection(s[0], ts, connectionCal, sqlcursor,spy,
                                   debug=debug,j=j,addRangeOpens=False,
                        training_dir='models/',training_name='stockEarningsModelTestv2noEPS',draw=True)
        #stock,j=ConfigTable(s[0], sqlcursor,ts,readType, j)
        if len(stock)==0:
            continue
        #try:
        #    if loadFromPickle and os.path.exists("%s.p" %s[0]):
        #        stock = pickle.load( open( "%s.p" %s[0], "rb" ) )
        #    else:
        #        stock=runTickerAlpha(ts,s[0])
        #        pickle.dump( stock, open( "%s.p" %s[0], "wb" ) )
        #        j+=1
        #except ValueError:
        #    print('ERROR processing stock...ValueError %s' %s[0])
        #    j+=1
        #    continue
        stockInput = formatInput(stock, s[0],spy_info, spy=spy)
        if stockInput!=None:
            entries+=[stockInput]
        del stock
    b.makeHTMLTable(outdir+'stockinfo.html',columns=columns,entries=entries)

# build the sector ETFs
columns=['Description']+columns
j=0
entries=[]
entries+=[['SPY']+formatInput(spy, 'SPY',spy_info,spy=spy)]
for s in b.etfs:
    if s[0]=='SPY':
        continue
    if j%n_ALPHA_PREMIUM_WAIT_ITER==0 and j!=0:
        time.sleep(56)
    #if j>1:
    #    break
    print(s[0])
    sys.stdout.flush()
    stock=None
    stock,j=GetNNSelection(s[0], ts,connectionCal, sqlcursor, spy,
                                   debug=debug,j=j,addRangeOpens=False,
                        training_dir='models/',
                        training_name='stockEarningsModelTestv2noEPS')
    #stock,j=ConfigTable(s[0], sqlcursor,ts,readType, j)
    if len(stock)==0:
        continue
    #try:
    #    if loadFromPickle and os.path.exists("%s.p" %s[0]):
    #        stock = pickle.load( open( "%s.p" %s[0], "rb" ) )
    #    else:
    #        stock=runTickerAlpha(ts,s[0])
    #        pickle.dump( stock, open( "%s.p" %s[0], "wb" ) )
    #        j+=1
    #except ValueError:
    #    print('ERROR processing...ValueError %s' %s[0])
    #    j+=1
    #    continue
    entries+=[[s[4]]+formatInput(stock, s[0],spy_info, spy=spy)]
    del stock;
b.makeHTMLTable(outdir+'sectorinfo.html',title='Sector Performance',columns=columns,entries=entries,linkIndex=1)
