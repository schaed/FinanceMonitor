from techindicators import techindicators
#import techindicators as techindicators
from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker,SQL_CURSOR,ConfigTable,GetTimeSlot,AddInfo,IS_ALPHA_PREMIUM_WAIT_ITER
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
import argparse
from zigzag import *

my_parser = argparse.ArgumentParser()
#my_parser.add_argument('--input', default='', type=str, required=True)
my_parser.add_argument('--filter', default='', type=str) # filter by ticker
my_parser.add_argument('--add', default='', type=str) # filter by ticker
args = my_parser.parse_args()

draw=False
outdir = b.outdir
doStocks=True
loadFromPickle=True
doETFs=True
doPDFs=False
debug=False
loadSQL=True
readType='full'

# https://www.investopedia.com/terms/z/zig_zag_indicator.asp
def plot_pivots(xaxis, yaxis, saveName='zigzag', xname='Date', yname='Beta',title='ZigZag'):
     """ Plotting the pivot points - when a stock changes direction
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         saveName : str
            Saved file name
         xname : str
            x-axis name
         yname : str
            y-axis name
         title : str
            Title of plot
     """
    #modes = pivots_to_modes(pivots) # 1 for valley to peak and -1 for peak to valley
    #pd.Series(X).pct_change().groupby(modes).describe().unstack()
    #compute_segment_returns(X, pivots)
    #max_drawdown(X) = (trough value - peak)/peak -> look for a small value

    plt.clf()
    pivots = peak_valley_pivots(yaxis.values, 0.05, -0.05)
    ts_pivots = yaxis #pd.Series(yaxis, index=xaxis)
    ts_pivots = ts_pivots[pivots != 0]
    yaxis.plot(color='black',alpha=0.8)
    
    plt.ylim(yaxis.min()*0.99, yaxis.max()*1.01)
    plt.plot(yaxis[pivots != 0].index, yaxis[pivots != 0], 'b-',alpha=0.5)
    plt.scatter(yaxis[pivots == 1].index, yaxis[pivots == 1], color='g')
    plt.scatter(yaxis[pivots == -1].index, yaxis[pivots == -1], color='r')

    plt.gcf().autofmt_xdate()
    #ts_pivots.plot(style='g-o');
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()
    
def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    """ Generic plotting with option to show support lines
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         xname : str
            x-axis name
         yname : str
            y-axis name
         saveName : str
            Saved file name
         hlines : array of horizontal lines drawn in matplotlib
         title : str
            Title of plot
         doSupport : bool
            Request generation of support lines on the fly
         my_stock_info : pandas data frame of stock timing and adj_close
     """
    # plotting
    plt.clf()
    plt.plot(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
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

def MakePlotMulti(xaxis, yaxis=[], colors=[], labels=[], xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    """ Generic plotting option for multiple plots 
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         xname : str
            x-axis name
         colors : array of strings
            colors compatible with matplotlib
         labels : array of str
            legend names
         yname : str
            y-axis name
         saveName : str
            Saved file name
         hlines : array of horizontal lines drawn in matplotlib
         title : str
            Title of plot    
         doSupport : bool
            Request generation of support lines on the fly
         my_stock_info : pandas data frame of stock timing and adj_close
    """
    # plotting
    j=0
    plt.clf()
    for y in yaxis:
        plt.plot(xaxis,y,color=colors[j],label=labels[j])
        j+=1
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    plt.legend(loc="upper left")
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()

# Draw the timing indices
def PlotTiming(data, ticker):
    """ Plot timing indicators including moving averages as well as Fourier Transforms
        
         Parameters:
         data - pandas data frame with time plus adj_close price
         ticker - string
                ticker symbol for the stock
    """
    if len(data)<1:
        return
    plt.clf()
    fig8 = plt.figure(constrained_layout=False)
    gs1 = fig8.add_gridspec(nrows=3, ncols=1, left=0.07, right=0.95, wspace=0.05)
    top_plt = fig8.add_subplot(gs1[:-1, :])
    top_plt.plot(data.index, data["adj_close"],color='black',label='Adj Close')
    top_plt.plot(data.index, data["sma200"],color='magenta',label='SMA200')
    top_plt.plot(data.index, data["sma100"],color='cyan',label='SMA100')
    top_plt.plot(data.index, data["sma50"],color='yellow',label='SMA50')
    top_plt.plot(data.index, data["sma20"],color='green',label='SMA20')
    top_plt.plot(data.index, data["ema13"],color='blue',label='EMA13')
    plt.legend(loc="upper center")
    techindicators.supportLevels(data)
    top_plt.grid(1)
    top_plt.set_ylabel('Closing Price')
    bottom_plt = fig8.add_subplot(gs1[-1, :])
    maxHT_DCPERIOD = max(max(data['HT_DCPERIOD']),1.0)/3.0
    maxHT_DCPHASE = max(max(data['HT_DCPHASE']),1.0)/3.0
    bottom_plt.plot(data.index, data['HT_DCPERIOD']/maxHT_DCPERIOD,color='red',label='Dominant Cycle Period')
    bottom_plt.plot(data.index, data['HT_DCPHASE']/maxHT_DCPHASE,color='green',label='Dominant Cycle Phase')
    bottom_plt.bar(data.index, data['HT_TRENDMODE'],color='blue',label='Trend vs Cycle Mode')
    bottom_plt.set_xlabel('%s Date' %ticker)
    bottom_plt.set_ylabel('Timing')
    plt.legend(loc="upper left")
    bottom_plt.grid(1)
    top_plt.set_title('HT Dominant cycle', fontsize=40)
    plt.gcf().set_size_inches(11,8)
    if doPDFs: plt.savefig(outdir+'timing_'+ticker+'.pdf')
    plt.savefig(outdir+'timing_'+ticker+'.png')

    top_plt.set_title('HTSine', fontsize=40)
    bottom_plt.clear()
    bottom_plt.plot(data.index, data['HT_SINE'],color='red',label='HTSine Slow')
    bottom_plt.plot(data.index, data['HT_SINElead'],color='green',label='HTSine Lead')
    bottom_plt.set_xlabel('%s Date' %ticker)
    bottom_plt.set_ylabel('HTSine')
    plt.legend(loc="upper left")
    bottom_plt.grid(1)

    plt.gcf().set_size_inches(11,8)
    if doPDFs: plt.savefig(outdir+'HTSine_'+ticker+'.pdf')
    plt.savefig(outdir+'HTSine_'+ticker+'.png')
    if not draw: plt.close()
        
# Draw the volume and the price by volume with various inputs
def PlotVolume(data, ticker):
    """ Plot Volume indicators along with moving averages 
        
         Parameters:
         data - pandas data frame with time plus adj_close price
         ticker - string
                ticker symbol for the stock
    """
    if len(data)<1:
        return
    # group the volume by closing price by the set division
    bucket_size = 0.025 * (max(data['adj_close']) - min(data['adj_close']))    
    volprofile  = data['volume'].groupby(data['adj_close'].apply(lambda x: bucket_size*round(x/bucket_size,0))).sum()/1.0e6
    posvolprofile  = data['pos_volume'].groupby(data['adj_close'].apply(lambda x: bucket_size*round(x/bucket_size,0))).sum()/1.0e6
    negvolprofile  = data['neg_volume'].groupby(data['adj_close'].apply(lambda x: bucket_size*round(x/bucket_size,0))).sum()/1.0e6
    
    plt.clf()
    fig8 = plt.figure(constrained_layout=False)
    gs1 = fig8.add_gridspec(nrows=3, ncols=1, left=0.07, right=0.95, wspace=0.05)
    top_plt = fig8.add_subplot(gs1[:-1, :])
    top_plt.set_title('Price by Volume',fontsize=40)    
    top_plt.plot(data.index, data["adj_close"],color='black',label='Adj Close')
    top_plt.plot(data.index, data["sma200"],color='magenta',label='SMA200')
    top_plt.plot(data.index, data["sma100"],color='cyan',label='SMA100')
    top_plt.plot(data.index, data["sma50"],color='yellow',label='SMA50')
    top_plt.plot(data.index, data["sma20"],color='peru',label='SMA20')
    top_plt.plot(data.index, data["ema13"],color='teal',label='EMA13')
    plt.legend(loc="upper center")
    techindicators.supportLevels(data)
    # normalize this bar chart to have half of the width. Need to get this because matplotlib doesn't use
    # timestamps. It converts them to an internal float.
    # Then we stack the negative and positive values
    xmin, xmax, ymin, ymax = top_plt.axis()
    normalize_vol=1.0
    try:
        normalize_vol = 0.5*(xmax - xmin)/max(volprofile.values)
    except:
        pass
    volprofile*=normalize_vol;   posvolprofile*=normalize_vol;     negvolprofile*=normalize_vol;
    plt.barh(posvolprofile.index.values, posvolprofile.values, height=0.9*bucket_size, align='center', color='green', alpha=0.45,left=xmin+(xmax-xmin)*0.001)
    leftStart = np.full(len(posvolprofile), xmin+(xmax-xmin)*0.001) + posvolprofile.values
    plt.barh(negvolprofile.index.values, negvolprofile.values, height=0.9*bucket_size, align='center', color='red', alpha=0.45,left=leftStart)
    top_plt.grid(1)
    top_plt.set_ylabel('Closing Price')
    bottom_plt = fig8.add_subplot(gs1[-1, :])
    bottom_plt.bar(data.index, data['neg_volume'],color='red')
    bottom_plt.bar(data.index, data['pos_volume'],color='green')
    bottom_plt.set_xlabel('%s Trading Volume' %ticker)
    bottom_plt.set_ylabel('Volume')
    bottom_plt.grid(1)

    plt.gcf().set_size_inches(11,8)
    if doPDFs: plt.savefig(outdir+'vol_'+ticker+'.pdf')
    plt.savefig(outdir+'vol_'+ticker+'.png')
    if not draw: plt.close()

def CandleStick(data, ticker):
    """ Plot candle stock plot of stock prices along with Bolanger Bands and Keltner ranges
        
         Parameters:
         data - pandas data frame with time plus adj_close price
         ticker - string
                ticker symbol for the stock
    """
    # Extracting Data for plotting
    #data = pd.read_csv('candlestick_python_data.csv')
    df = data.loc[:, ['open', 'high', 'low', 'close','volume']]
    df.columns = ['Open', 'High', 'Low', 'Close','Volume']
    df['UpperB'] = data['BolUpper']        
    df['LowerB'] = data['BolLower']
    df['KeltLower'] = data['KeltLower']
    df['KeltUpper'] = data['KeltUpper']
    df['sma200'] = data['sma200']
    df['sma50'] = data['sma50']
    df['sma20'] = data['sma20']

    if len(df['Open'])<1:
        return
    # Plot candlestick.
    # Add volume.
    # Add moving averages: 3,6,9.
    # Save graph to *.png.
    ap0 = [ mpf.make_addplot(df['UpperB'],color='g',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default
        mpf.make_addplot(df['LowerB'],color='b',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default
        mpf.make_addplot(df['sma200'],color='r',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default        
        mpf.make_addplot(df['KeltLower'],color='darkviolet',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default
        mpf.make_addplot(df['KeltUpper'],color='magenta',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default
        mpf.make_addplot(df['sma50'],color='teal',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default        
        mpf.make_addplot(df['sma20'],color='peru',secondary_y=False,alpha=0.5,y_on_right=False),  # uses panel 0 by default        
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
    #for 
    axes[0].legend(['Price','Bolanger Up','Bolanger Down','SMA200','Kelt+','Kelt-','SMA50','SMA20'])
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
    """ Plot 5 year time window
        
         Parameters:
         my_stock_info - pandas data frame with time plus adj_close price
         market - pandas data frame with the S&P
         ticker - string
                ticker symbol for the stock
         plttext - string
                Label for the plot
    """
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
    plt.title('5y Return', fontsize=30)
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
    """ DrawPlots - Draw all plots of the stock along with market comparisons
        
         Parameters:
         my_stock_info - pandas data frame with time plus adj_close price
         market - pandas data frame with the S&P
         ticker - string
                ticker symbol for the stock
         plttext - string
                Common label for the plot
    """
    #plt.plot(stock_info.index,stock_info['close'])

    if not draw:
        plt.ioff()

    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['adj_close'],my_stock_info['sma50'],my_stock_info['sma200']],colors=['black','green','red'], labels=['Closing','SMA50','SMA200'], xname='Date',yname='Closing price',saveName='price_support%s_%s' %(plttext,ticker), doSupport=True,my_stock_info=my_stock_info,title='Support Lines')
    MakePlot(my_stock_info.index, my_stock_info['copp'], xname='Date',yname='Coppuck Curve',saveName='copp%s_%s' %(plttext,ticker),hlines=[(0.0,'black','-')],title='Coppuck Curve')
    MakePlot(my_stock_info.index, my_stock_info['sharpe'], xname='Date',yname='Sharpe Ratio',saveName='sharpe%s_%s' %(plttext,ticker),title='Sharpe Ratio')
    MakePlot(my_stock_info.index, my_stock_info['beta'], xname='Date',yname='Beta',saveName='beta%s_%s' %(plttext,ticker),title='Beta')
    MakePlot(my_stock_info.index, my_stock_info['alpha'], xname='Date',yname='Alpha',saveName='alpha%s_%s' %(plttext,ticker), hlines=[(0.0,'black','-')],title='Alpha')
    MakePlot(my_stock_info.index, my_stock_info['adx'], xname='Date',yname='ADX',saveName='adx%s_%s' %(plttext,ticker), hlines=[(25.0,'black','dotted')],title='ADX')
    MakePlot(my_stock_info.index, my_stock_info['willr'], xname='Date',yname='Will %R',saveName='willr%s_%s' %(plttext,ticker), hlines=[(-20.0,'red','dotted'),(-80.0,'green','dotted')],title='Will %R')
    MakePlot(my_stock_info.index, my_stock_info['ultosc'], xname='Date',yname='Ultimate Oscillator',saveName='ultosc%s_%s' %(plttext,ticker), hlines=[(30.0,'green','dotted'),(70.0,'green','dotted')],title='Ultimate Oscillator')
    MakePlot(my_stock_info.index, my_stock_info['rsquare'], xname='Date',yname='R-squared',saveName='rsquare%s_%s' %(plttext,ticker), hlines=[(0.7,'black','-')],title='R-squared')
    MakePlot(my_stock_info.index, my_stock_info['cmf'], xname='Date',yname='CMF',saveName='cmf%s_%s' %(plttext,ticker), hlines=[(0.2,'green','dotted'),(0.0,'black','-'),(-0.2,'red','dotted')],title='Chaikin Money Flow')
    MakePlot(my_stock_info.index, my_stock_info['mfi_bill_ana'], xname='Date',yname='MFI = 4 is buy',saveName='mfi_bill_ana%s_%s' %(plttext,ticker), hlines=[(4.0,'green','dotted'),(0.0,'black','dotted'),(3.0,'red','dotted')],title='Market Fluctuation index')
    MakePlot(my_stock_info.index, my_stock_info['mfi'], xname='Date',yname='MFI',saveName='mfi%s_%s' %(plttext,ticker), hlines=[(20.0,'green','dotted'),(50.0,'black','-'),(80.0,'red','dotted')],title='Money Flow Index')
    MakePlot(my_stock_info.index, my_stock_info['cci'], xname='Date',yname='Commodity Channel Index',saveName='cci%s_%s' %(plttext,ticker),title='Commodity Channel Index')
    MakePlot(my_stock_info.index, my_stock_info['obv'], xname='Date',yname='On Balanced Volume',saveName='obv%s_%s' %(plttext,ticker),title='On Balanced Volume')    
    MakePlot(my_stock_info.index, my_stock_info['force'], xname='Date',yname='Force Index',saveName='force%s_%s' %(plttext,ticker),title='Force Index')
    MakePlot(my_stock_info.index, my_stock_info['bop'], xname='Date',yname='Balance of Power',saveName='bop%s_%s' %(plttext,ticker),  hlines=[(0.0,'black','dotted')],title='Balance of Power')
    MakePlot(my_stock_info.index, my_stock_info['chosc'], xname='Date',yname='Chaikin Oscillator',saveName='chosc%s_%s' %(plttext,ticker),title='Chaikin Oscillator')
    MakePlot(my_stock_info.index, my_stock_info['corr14'], xname='Date',yname='14d Correlation with SPY',saveName='corr%s_%s' %(plttext,ticker),hlines=[(0.0,'black','dotted')],title='14d Correlation with SPY')

    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['macd'],my_stock_info['macdsignal']], colors=['red','blue'], labels=['MACD','Signal'], xname='Date',yname='MACD',saveName='macd%s_%s' %(plttext,ticker),title='MACD')
    if 'aroon' in my_stock_info:
        MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['aroonUp'],my_stock_info['aroonDown']], colors=['red','blue'], labels=['Up','Down'], xname='Date',yname='AROON',saveName='aroon%s_%s' %(plttext,ticker),title='AROON')
    if 'jaws' in my_stock_info:
        MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['adj_close'],my_stock_info['jaws'],my_stock_info['teeth'],my_stock_info['lips']], colors=['black','blue','red','green'], labels=['Closing','Jaws','teeth','lips'], xname='Date',yname='Closing Price',saveName='alligator%s_%s' %(plttext,ticker),title='Alligator')
    if 'bullPower' in my_stock_info:
        MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['bullPower'],my_stock_info['bearPower']], colors=['green','red'], labels=['Bull Power','Bear Power'], xname='Date',yname='Bull/Bear Power',saveName='bullbear%s_%s' %(plttext,ticker),title='Bull/Bear Power')
    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['adj_close'],my_stock_info['vwap10'],my_stock_info['vwap14'],my_stock_info['vwap20']], colors=['red','blue','green','magenta'], labels=['Close Price','VWAP10','VWAP14','VWAP20'], xname='Date',yname='Price',saveName='vwap10%s_%s' %(plttext,ticker),title='Volume Weighted AP')
    MakePlotMulti(my_stock_info.index, yaxis=[my_stock_info['stochK'],my_stock_info['stochD']], colors=['red','blue'], labels=['%K','%D'], hlines=[(80.0,'green','dotted'),(20.0,'red','dotted')], xname='Date',yname='Price',saveName='stoch%s_%s' %(plttext,ticker),title='Stochastic')

    # plot volume
    PlotVolume(my_stock_info, ticker)
    # Plot timing indicators
    PlotTiming(my_stock_info, ticker)

    # plot the zigzag
    start = time.time()
    plot_pivots(my_stock_info.index, my_stock_info.adj_close, saveName='zigzag%s_%s' %(plttext,ticker), xname='Date', yname='Closing Price')
    end = time.time()
    if debug: print('Process time to find pivots: %s' %(end - start))
    
    # plot Ichimoku Cloud
    plt.cla()
    # Plot closing price and parabolic SAR
    komu_cloud = my_stock_info[['adj_close','SAR']].plot()
    plt.ylabel('Price')
    plt.xlabel('Date')
    komu_cloud.fill_between(my_stock_info.index, my_stock_info.senkou_spna_A, my_stock_info.senkou_spna_B,
    where=my_stock_info.senkou_spna_A >= my_stock_info.senkou_spna_B, color='lightgreen')
    komu_cloud.fill_between(my_stock_info.index, my_stock_info.senkou_spna_A, my_stock_info.senkou_spna_B,
    where=my_stock_info.senkou_spna_A < my_stock_info.senkou_spna_B, color='lightcoral')
    plt.grid(True)
    plt.legend()
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'komu%s_%s.pdf' %(plttext,ticker))
    plt.savefig(outdir+'komu%s_%s.png' %(plttext,ticker))
    if not draw: plt.close()
    
    # comparison to the market
    plt.plot(my_stock_info.index,my_stock_info['yearly_return'],color='blue',label=ticker)    
    plt.plot(market.index,     market['yearly_return'],   color='red', label='SPY')    
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.ylabel('Yearly Return')
    plt.xlabel('Date')
    plt.legend(loc="upper left")
    plt.title('Yearly Return',fontsize=30)
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
    # collect all of the chart signals
    chartSignals = []
    if len(my_stock_info)>0:
        for col in my_stock_info.columns:
            if 'CDL' in col:
                if my_stock_info[col][-1]>0:
                    chartSignals+=[[col[3:],1,0,0]]
                    if debug: print('%s signal buy: %s' %(col,ticker))
                    if col.count('CDLABANDONEDBABY'):  print('%s Abandoned Baby by signal: %s' %(col,ticker))
                if my_stock_info[col][-2]>0:
                    if debug: print('%s signal buy 2days ago: %s' %(col,ticker))
                    chartSignals+=[[col[3:],0,1,0]]
                if my_stock_info[col][-5:].sum()>0:
                    if debug: print('%s signal buy in last 5 days: %s' %(col,ticker))
                    chartSignals+=[[col[3:],0,0,1]]
    return chartSignals
        #if my_stock_info['CDLHAMMER'][-1]>0:
        #    print('Abandoned baby signal buy: %s' %ticker)
    
def SARTradingStategy(stock):
    """ SARTradingStategy - Back testing of the SAR trading model. Shows the cumulative return from the strategy
        
         Parameters:
         stock - pandas data frame with time plus adj_close price
    """
    # Trade strategy from SAR
    stock['signal'] = 0
    stock.loc[(stock.close > stock.senkou_spna_A) & (stock.close >stock.senkou_spna_B) & (stock.close > stock.SAR), 'signal'] = 1
    stock.loc[(stock.close < stock.senkou_spna_A) & (stock.close < stock.senkou_spna_B) & (stock.close < stock.SAR), 'signal'] = -1
    stock['signal'].value_counts()
    # Calculate daily returns
    daily_returns = stock.Close.pct_change()
    # Calculate strategy returns
    strategy_returns = daily_returns *stock['signal'].shift(1)
    # Calculate cumulative returns
    (strategy_returns+1).cumprod().plot(figsize=(10,5))
    # Plot the strategy returns
    plt.xlabel('Date')
    plt.ylabel('Strategy Returns (%)')
    plt.grid()
    plt.show()
        
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
#if loadFromPickle and os.path.exists("%s.p" %ticker):
#    stock_info = pickle.load( open( "%s.p" %ticker, "rb" ) )
#    #spy = pickle.load( open( "SPY.p", "rb" ) )
#    #spy.to_sql('SPY',sqlcursor,if_exists='append',index=True)
#else:
#    #stock_info = runTicker(api,ticker)
#    stock_info=runTickerAlpha(ts,ticker,readType)
#    spy=runTickerAlpha(ts,'SPY',readType)
#    pickle.dump( spy, open( "SPY.p", "wb" ) )
#    pickle.dump( stock_info, open( "%s.p" %ticker, "wb" ) )
# add info
if len(spy)==0:
    print('ERROR - empy info %s' %ticker)
spy['daily_return']=spy['adj_close'].pct_change(periods=1)
AddInfo(spy, spy)
spy_1year = GetTimeSlot(spy)
DrawPlots(spy_1year,'SPY',spy_1year)
n_ALPHA_PREMIUM_WAIT_ITER = IS_ALPHA_PREMIUM_WAIT_ITER()
j=0
cdir = os.getcwd()
if doStocks:
    runList =  b.stock_list
    if args.filter!='':
        runList+=[[args.filter,0,0,'NYSE','']]
    if args.add!='':
        addt = args.add.split(',')
        runList=[]        
        for at in addt:
            runList+=[[at,0,0,'NYSE','']]
    for s in runList:
        if args.filter!='':
            if s[0]!=args.filter:
                continue
        if s[0]=='SPY':
            continue
        if s[0].count('^'):
            continue
        if j%n_ALPHA_PREMIUM_WAIT_ITER==0 and j!=0:
            time.sleep(56)
        print(s[0])
        sys.stdout.flush()
        
        tstock_info,j=ConfigTable(s[0], sqlcursor,ts,readType, j)

        if len(tstock_info)==0:
            continue
        # draw before we shorten this to 1 year
        LongTermPlot(tstock_info,spy,ticker=s[0])
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
        print(tstock_info)
        try:
            start = time.time()
            AddInfo(tstock_info, spy)
            end = time.time()
            if debug: print('Process time to add info: %s' %(end - start))
        except (ValueError,KeyError):
            print('Error processing %s' %s[0])
            j+=1
            continue
                
        tstock_info = GetTimeSlot(tstock_info) # gets the one year timeframe
        start = time.time()
        chartSignals =DrawPlots(tstock_info,s[0],spy_1year)
        if len(chartSignals)>0.0:
            chartSignals = pd.DataFrame(chartSignals,columns=['Chart Signal', 'Yesterday','Two Days Ago','In Last 5 days'])
            chartSignals = chartSignals.groupby(chartSignals['Chart Signal']).sum().reset_index()

        end = time.time()
        print('Process time to add draw: %s' %(end - start))
        os.chdir(outdir)
        b.makeHTML('%s.html' %s[0],s[0],filterPattern='*_%s' %s[0],describe=s[4], chartSignals=chartSignals)
        os.chdir(cdir)    
        del tstock_info;
if doETFs:
    j=0
    for s in b.etfs:
        if s[0].count('^'):
            continue
        if j%n_ALPHA_PREMIUM_WAIT_ITER==0 and j!=0:
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
