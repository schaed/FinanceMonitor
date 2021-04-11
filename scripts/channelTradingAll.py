from techindicators import techindicators
#import techindicators as techindicators
from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker,SQL_CURSOR,ConfigTable,GetTimeSlot
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
import matplotlib
matplotlib.use('Agg') 
import mplfinance as mpf
draw=False
outdir = b.outdir
doStocks=True
loadFromPickle=True
doETFs=True
doPDFs=False
debug=False
loadSQL=True
readType='full'
def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    # plotting
    plt.clf()
    plt.plot(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()

def MakePlotMulti(xaxis, yaxis=[], colors=[], labels=[], xname='Date',yname='Beta',saveName='', hlines=[],title=''):
    # plotting
    j=0
    for y in yaxis:
        plt.plot(xaxis,y,color=colors[j],label=labels[j])
        j+=1
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    plt.legend(loc="upper left")
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    
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

    if len(df['Open'])<1:
        return
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
    fig,axes=mpf.plot(df, type='candle', style='charles',
            title=ticker,
            ylabel='Price ($) %s' %ticker,
            ylabel_lower='Shares \nTraded',
            volume=True, 
            mav=(200),
            addplot=ap0,
            returnfig=True) #,
            #savefig=outdir+'test-mplfiance_'+ticker+'.png')
    

        # Configure chart legend and title
    axes[0].legend(['Price','Bolanger Up','Bolanger Down','SMA200','Kelt+','Kelt-'])
    #axes[0].set_title(ticker)
    # Save figure to file
    if doPDFs: fig.savefig(outdir+'test-mplfiance_'+ticker+'.pdf')
    fig.savefig(outdir+'test-mplfiance_'+ticker+'.png')
    techindicators.plot_support_levels(ticker,df,[mpf.make_addplot(df['sma200'],color='r') ],outdir=outdir,doPDF=doPDFs)
    del fig
    del axes
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
    
def LongTermPlot(my_stock_info,market,ticker,plttext=''):
    date_diff = 5*365
    my_stock_info5y = GetTimeSlot(my_stock_info, days=date_diff)
    market5y = GetTimeSlot(market, days=date_diff)
    min_length = min(len(my_stock_info5y),len(market5y))
    max_length = max(len(my_stock_info5y),len(market5y))
    if min_length<max_length:
        my_stock_info5y = my_stock_info5y[-min_length:]
        market5y = market5y[-min_length:]

    if len(market5y['adj_close'])<1 or len(my_stock_info5y['adj_close'])<1:
        print('Ticker has no adjusted close info: %s' %ticker)
        return
    my_stock_info5y['year5_return']=my_stock_info5y['adj_close']/my_stock_info5y['adj_close'][0]-1
    market5y['year5_return']=market5y['adj_close']/market5y['adj_close'][0]-1
    # comparison to the market
    plt.clf()
    plt.plot(my_stock_info5y.index,my_stock_info5y['year5_return'],color='blue',label=ticker)    
    plt.plot(market5y.index,     market5y['year5_return'],   color='red', label='SPY')    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('5 Year Return')
    plt.xlabel('Date')
    plt.legend(loc="upper left")
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'longmarket%s_%s.pdf' %(plttext,ticker))
    plt.savefig(outdir+'longmarket%s_%s.png' %(plttext,ticker))
    if not draw: plt.close()

def DrawPlots(my_stock_info,ticker,market,plttext=''):
    #plt.plot(stock_info.index,stock_info['close'])

    if not draw:
        plt.ioff()
    MakePlot(my_stock_info.index, my_stock_info['adj_close'], xname='Date',yname='Closing price',saveName='price_support%s_%s' %(plttext,ticker), doSupport=True,my_stock_info=my_stock_info)
    MakePlot(my_stock_info.index, my_stock_info['copp'], xname='Date',yname='Coppuck Curve',saveName='copp%s_%s' %(plttext,ticker),hlines=[(0.0,'black','-')])
    MakePlot(my_stock_info.index, my_stock_info['sharpe'], xname='Date',yname='Sharpe Ratio',saveName='sharpe%s_%s' %(plttext,ticker))
    MakePlot(my_stock_info.index, my_stock_info['beta'], xname='Date',yname='Beta',saveName='beta%s_%s' %(plttext,ticker))
    MakePlot(my_stock_info.index, my_stock_info['alpha'], xname='Date',yname='Alpha',saveName='beta%s_%s' %(plttext,ticker), hlines=[(0.0,'black','-')],title=' Alpha')
    MakePlot(my_stock_info.index, my_stock_info['rsquare'], xname='Date',yname='R-squared',saveName='rsquare%s_%s' %(plttext,ticker), hlines=[(0.7,'black','-')])
    MakePlot(my_stock_info.index, my_stock_info['cmf'], xname='Date',yname='CMF',saveName='cmf%s_%s' %(plttext,ticker), hlines=[(0.2,'green','dotted'),(0.0,'black','-'),(-0.2,'red','dotted')])
    MakePlot(my_stock_info.index, my_stock_info['cci'], xname='Date',yname='Commodity Channel Index',saveName='cci%s_%s' %(plttext,ticker))
    MakePlot(my_stock_info.index, my_stock_info['obv'], xname='Date',yname='On Balanced Volume',saveName='obv%s_%s' %(plttext,ticker))    
    MakePlot(my_stock_info.index, my_stock_info['force'], xname='Date',yname='Force Index',saveName='force%s_%s' %(plttext,ticker))
    MakePlot(my_stock_info.index, my_stock_info['chosc'], xname='Date',yname='Chaikin Oscillator',saveName='chosc%s_%s' %(plttext,ticker))

    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['macd'],my_stock_info['macdsignal']], colors=['red','blue'], labels=['MACD','Signal'], xname='Date',yname='MACD',saveName='macd%s_%s' %(plttext,ticker))
    if 'aroon' in my_stock_info:
        MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['aroonUp'],my_stock_info['aroonDown']], colors=['red','blue'], labels=['Up','Down'], xname='Date',yname='AROON',saveName='aroon%s_%s' %(plttext,ticker))        
    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['adj_close'],my_stock_info['vwap10'],my_stock_info['vwap14'],my_stock_info['vwap20']], colors=['red','blue','green','magenta'], labels=['Close Price','VMAP10','VMAP14','VMAP20'], xname='Date',yname='Price',saveName='vwap10%s_%s' %(plttext,ticker))
    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['stochK'],my_stock_info['stochD']], colors=['red','blue'], labels=['%K','%D'], hlines=[(80.0,'green','dotted'),(20.0,'red','dotted')], xname='Date',yname='Price',saveName='stoch%s_%s' %(plttext,ticker))     

    # comparison to the market
    plt.plot(my_stock_info.index,my_stock_info['yearly_return'],color='blue',label=ticker)    
    plt.plot(market.index,     market['yearly_return'],   color='red', label='SPY')    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Yearly Return')
    plt.xlabel('Date')
    plt.legend(loc="upper left")
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'market%s_%s.pdf' %(plttext,ticker))
    plt.savefig(outdir+'market%s_%s.png' %(plttext,ticker))
    if not draw: plt.close()
    # comparison to the market monthly returns
    plt.bar(my_stock_info['monthly_return'].dropna().index,my_stock_info['monthly_return'].dropna(),color='blue',label=ticker,width = 5.25)    
    plt.bar(market['monthly_return'].dropna().index,     market['monthly_return'].dropna(),   color='red', label='SPY', width = 5.25)
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Monthly Return')
    plt.xlabel('Date')
    plt.legend(loc="upper left")
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'monthlymarket%s_%s.pdf' %(plttext,ticker))
    plt.savefig(outdir+'monthlymarket%s_%s.png' %(plttext,ticker))
    if not draw: plt.close()
        
    CandleStick(my_stock_info,ticker)
    
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
    start = time.time()
    stock['KeltLower'],stock['KeltCenter'],stock['KeltUpper']=techindicators.kelt(stock['high'],stock['low'],stock['close'],20,2.0,20)
    stock['copp']=techindicators.copp(stock['close'],14,11,10)
    stock['daily_return']=stock['adj_close'].pct_change(periods=1)
    stock['daily_return_stddev14']=techindicators.rstd(stock['daily_return'],14)
    stock['beta']=techindicators.rollingBetav2(stock,14,market)
    stock['alpha']=techindicators.rollingAlpha(stock,14,market)
    stock['rsquare']=techindicators.rollingRsquare(stock,14,market)
    stock['sharpe']=techindicators.sharpe(stock['daily_return'],30) # generally above 1 is good
    start = time.time()
    stock['cci']=techindicators.cci(stock['high'],stock['low'],stock['close'],20) 
    stock['stochK'],stock['stochD']=techindicators.stoch(stock['high'],stock['low'],stock['close'],14,3,3)    
    stock['obv']=techindicators.obv(stock['adj_close'],stock['volume'])
    stock['force']=techindicators.force(stock['adj_close'],stock['volume'],13)
    stock['macd'],stock['macdsignal']=techindicators.macd(stock['adj_close'],12,26,9)
    #stock['pdmd'],stock['ndmd'],stock['adx']=techindicators.adx(stock['high'],stock['low'],stock['close'],14)
    stock['aroonUp'],stock['aroonDown'],stock['aroon']=techindicators.aroon(stock['high'],stock['low'],25)
    stock['vwap14']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],14)
    stock['vwap10']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],10)
    stock['vwap20']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],20)
    stock['chosc']=techindicators.chosc(stock['high'],stock['low'],stock['close'],stock['volume'],3,10)
    end = time.time()
    if debug: print('Process time to new: %s' %(end - start))
    stock['weekly_return']=stock['adj_close'].pct_change(freq='W')
    stock['monthly_return']=stock['adj_close'].pct_change(freq='M')
    stock_1y = GetTimeSlot(stock)
    if len(stock_1y['adj_close'])<1:
        print('Ticker has no adjusted close info: %s' %ticker)
        stock['yearly_return']=stock['adj_close']
    else:
        stock['yearly_return']=stock['adj_close']/stock_1y['adj_close'][0]-1

api = ALPACA_REST()
ts = ALPHA_TIMESERIES()
spy = runTicker(api,'SPY')
ticker='TSLA'
#ticker='TSLA'
stock_info=None
spy=None
sqlcursor = SQL_CURSOR()
spy,j = ConfigTable('SPY', sqlcursor,ts,readType)
print('spy')
print(spy)
if loadFromPickle and os.path.exists("%s.p" %ticker):
    stock_info = pickle.load( open( "%s.p" %ticker, "rb" ) )
    #spy = pickle.load( open( "SPY.p", "rb" ) )
    #spy.to_sql('SPY',sqlcursor,if_exists='append',index=True)
else:
    #stock_info = runTicker(api,ticker)
    stock_info=runTickerAlpha(ts,ticker,readType)
    spy=runTickerAlpha(ts,'SPY',readType)
    pickle.dump( spy, open( "SPY.p", "wb" ) )
    pickle.dump( stock_info, open( "%s.p" %ticker, "wb" ) )
# add info
if len(stock_info)==0:
    print('ERROR - empy info %s' %ticker)
spy['daily_return']=spy['adj_close'].pct_change(periods=1)
AddInfo(spy, spy)
spy_1year = GetTimeSlot(spy)
DrawPlots(spy_1year,'SPY',spy_1year)

j=0
cdir = os.getcwd()
if doStocks:
    for s in b.stock_list:
        #if s[0]!='S':
        #    continue
        if s[0]=='SPY':
            continue
        if s[0].count('^'):
            continue
        if j%4==0 and j!=0:
            time.sleep(56)
        print(s[0])
        sys.stdout.flush()
        
        tstock_info,j=ConfigTable(s[0], sqlcursor,ts,readType, j)
        if len(tstock_info)==0:
            continue
        #if j>2:
        #    break
        #try:
        #    if loadFromPickle and os.path.exists("%s.p" %s[0]):
        #        start = time.time()
        #        tstock_info = pickle.load( open( "%s.p" %s[0], "rb" ) )
        #        tstock_info.to_sql(s[0],sqlcursor,if_exists='append',index=True)
        #        end = time.time()
        #        if debug: print('Process time to load file: %s' %(end - start))
        #    else:
        #        tstock_info=runTickerAlpha(ts,s[0],readType)
        #        pickle.dump( tstock_info, open( "%s.p" %s[0], "wb" ) )
        #        tstock_info.to_sql(s[0],sqlcursor,if_exists='append',index=True)
        #        j+=1
        #except ValueError:
        #    print('ERROR processing...ValueError %s' %s[0])
        #    j+=1
        #    continue
        try:
            start = time.time()
            AddInfo(tstock_info, spy)
            end = time.time()
            if debug: print('Process time to add info: %s' %(end - start))
        except ValueError:
            print('Error processing %s' %s[0])
            j+=1
            continue
        tstock_info = GetTimeSlot(tstock_info) # gets the one year timeframe
        start = time.time()
        DrawPlots(tstock_info,s[0],spy_1year)
        LongTermPlot(tstock_info,spy,ticker=s[0])
        end = time.time()
        print('Process time to add draw: %s' %(end - start))
        os.chdir(outdir)
        b.makeHTML('%s.html' %s[0],s[0],filterPattern='*_%s' %s[0],describe=s[4])
        os.chdir(cdir)    
        del tstock_info;
if doETFs:
    j=0
    for s in b.etfs:
        if s[0].count('^'):
            continue
        if j%4==0 and j!=0:
            time.sleep(56)
        print(s[0])
        sys.stdout.flush()
        estock_info=None
        estock_info,j=ConfigTable(s[0], sqlcursor,ts,readType, j)
        if len(estock_info)==0:
            continue
        #try:
        #    if loadFromPickle and os.path.exists("%s.p" %s[0]):
        #        start = time.time()
        #        estock_info = pickle.load( open( "%s.p" %s[0], "rb" ) )
        #        estock_info.to_sql(s[0],sqlcursor,if_exists='append',index=True)
        #        end = time.time()
        #        if debug: print('Process time to load file: %s' %(end - start))
        #    else:
        #        estock_info=runTickerAlpha(ts,s[0],readType)
        #        pickle.dump( estock_info, open( "%s.p" %s[0], "wb" ) )
        #        estock_info.to_sql(s[0],sqlcursor,if_exists='append',index=True)
        #        j+=1
        #except ValueError:
        #    print('ERROR processing...ValueError %s' %s[0])
        #    j+=1
        #    continue
        LongTermPlot(estock_info,spy,ticker=s[0])

        try:
            start = time.time()
            AddInfo(estock_info, spy)
            end = time.time()
            if debug: print('Process time to add info: %s' %(end - start))
        except ValueError:
            print('Error processing %s' %s[0])
            continue
        start = time.time()
        estock_info = GetTimeSlot(estock_info) # gets the one year timeframe
        end = time.time()
        if debug: print('Process time to get time slot: %s' %(end - start))
        start = time.time()
        DrawPlots(estock_info,s[0],spy_1year)
        end = time.time()
        if debug: print('Process time to draw plots: %s' %(end - start))
        os.chdir(outdir)
        b.makeHTML('%s.html' %s[0],s[0],filterPattern='*_%s' %s[0],describe=s[4],linkIndex=0)
        os.chdir(cdir)
        del estock_info
