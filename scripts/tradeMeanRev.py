import time,os,sys
import copy
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
from matplotlib.backends.backend_agg import RendererAgg

# collect data
from ReadData import ALPACA_REST,runTicker,ALPHA_TIMESERIES,SQL_CURSOR,ConfigTable,GetTimeSlot,ALPHA_FundamentalData,AddSMA,AddInfo
from alpaca_trade_api.rest import TimeFrame
import alpaca_trade_api
import statsmodels.api as sm1
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from Earnings import GetIncomeStatement,GetPastEarnings,GetStockOverview,GetBalanceSheetQuarterly,GetBalanceSheetAnnual
import base

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import matplotlib.dates as mdates

import pytz
import datetime
est = pytz.timezone('US/Eastern')
api = ALPACA_REST()
ts = ALPHA_TIMESERIES()
STOCK_DB_PATH = os.getenv('STOCK_DB_PATH')
sqlcursor = SQL_CURSOR('%s/stocksAV.db' %STOCK_DB_PATH)

matplotlib.use("agg")
sns.set_style('darkgrid')

def FitWithBand(my_index, arr_prices, doMarker=True, ticker='X',outname='', poly_order = 2, price_key='adj_close',doDateKey=False,spy_comparison=[], doRelative=False, doPlot=False, doJoin=True, debug=False):
    """
    my_index : datetime array
    price : array of prices or values
    doMarker : bool : draw markers or if false draw line
    ticker : str : ticker symbol name
    outname : str : name of histogram to save as
    poly_order : int : order of polynomial to fit
    price_key : str : name of the price to entry to fit
    spy_comparison : array : array of prices to use as a reference. don't use when None
    doRelative : bool : compute the error bands with relative changes. Bigger when there is a big change in price
    doJoin : bool : join the two arrays on matching times
"""
    prices = arr_prices[price_key]
    x=my_index.values
    #print(x)
    if not doDateKey:
        x = mdates.date2num(my_index)
    xx = x
    dd = xx
    if not doDateKey:
        xx = np.linspace(x.min(), x.max(), 1000)        
        dd = mdates.num2date(xx)

    # prepare a spy comparison
    ylabelPlot='Price for %s' %ticker 
    if len(spy_comparison)>0:
        ylabelPlot='Price for %s / SPY' %ticker
           
        if not doJoin:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.loc[arr_prices.index,:]
            prices /= (spy_comparison[price_key] / spy_comparison[price_key][-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'high'] /= (spy_comparison.high / spy_comparison.high[-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'low']  /= (spy_comparison.low  / spy_comparison.low[-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'open'] /= (spy_comparison.open / spy_comparison.open[-1])
        else:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.copy(True)
            arr_prices = arr_prices.join(spy_comparison,how='left',rsuffix='_spy')
            for i in ['high','low','open','close',price_key]:
                if len(arr_prices[i+'_spy'])>0:
                    arr_prices[i] /= (arr_prices[i+'_spy'] / arr_prices[i+'_spy'][-1])
            prices = arr_prices[price_key]

    # perform the fit
    #print(x,prices)
    z4 = np.polyfit(x, prices, poly_order)
    p4 = np.poly1d(z4)

    # create an error band
    diff = prices - p4(x)
    stddev = diff.std()

    output_lines = '%s,%s,%s,%s' %(p4(x)[-1],stddev,diff[-1],prices[-1])
    output_linesn = [p4(x)[-1],stddev,diff[-1],prices[-1],p4,x[-1]]
    if stddev!=0.0:
        output_lines = '%0.3f,%0.3f,%0.3f,%s' %(p4(x)[-1],stddev,diff[-1]/stddev,prices[-1])
        output_linesn = [p4(x)[-1],stddev,diff[-1]/stddev,prices[-1],p4,x[-1]]
    if doRelative:
        diff /= p4(x)
        stddev = diff.std() #*p4(x).mean()
        
    pos_count_1sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-0.5*stddev))))
    pos_count_2sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-1.0*stddev))))
    pos_count_3sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-1.5*stddev))))
    pos_count_4sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-2.0*stddev))))
    pos_count_5sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-2.5*stddev))))
    if len(diff)>0:
        if debug: print('Time period: %s for ticker: %s' %(outname,ticker))
        coverage_txt='Percent covered\n'
        coverage = [100.0*pos_count_1sigma/len(diff) , 100.0*pos_count_2sigma/len(diff),
                        100.0*pos_count_3sigma/len(diff) , 100.0*pos_count_4sigma/len(diff),
                        100.0*pos_count_5sigma/len(diff) ]
        for i in range(0,5):
            if debug: print('Percent outside %i std. dev.: %0.2f' %(i+1,coverage[i]))
            coverage_txt+='%i$\sigma$: %0.1f\n' %(i+1,coverage[i])

    if not doPlot:
        return output_linesn
    
    if doRelative:
        stddev *= p4(x).mean()
    smaPlots=[]
    if  len(spy_comparison)==0 and 'sma200' in arr_prices.columns:
        smaPlots = [go.Scatter(
            name='SMA200',
            x=my_index,
            y=arr_prices['sma200'],
            mode='lines',
            line=dict(color='rgb(500, 119, 0)')),
                    go.Scatter(
            name='SMA100',
            x=my_index,
            y=arr_prices['sma100'],
            mode='lines',
                        line=dict(color='rgb(7, 119, 0)')),
                    go.Scatter(
            name='SMA50',
            x=my_index,
            y=arr_prices['sma50'],
            mode='lines',
                        line=dict(color='rgb(100, 9, 0)')),                    
        ]


        
    fig = go.Figure([
        go.Scatter(
            name='Price',
            x=my_index,
            y=p4(x),
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Adj Close',
            x=my_index,
            y=arr_prices.adj_close,
            mode='lines',
            line=dict(color='rgb(31, 219, 0)'),
        ),        
        go.Scatter(
            name='1 sigma',
            x=my_index,
            y=p4(x)+0.5*stddev,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ),
        go.Scatter(
            name='-1sigma',
            x=my_index,
            y=p4(x)-0.5*stddev,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ),        go.Scatter(
            name='2 sigma',
            x=my_index,
            y=p4(x)+1.0* stddev,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ),
        go.Scatter(
            name='-2sigma',
            x=my_index,
            y=p4(x)-1.0*stddev,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False
        ),        go.Scatter(
            name='3 sigma',
            x=my_index,
            y=p4(x)+1.5* stddev,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ),
        go.Scatter(
            name='-3sigma',
            x=my_index,
            y=p4(x)-1.5*stddev,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False
        ),        go.Scatter(
            name='4 sigma',
            x=my_index,
            y=p4(x)+2.0* stddev,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=True
        ),
        go.Scatter(
            name='-4sigma',
            x=my_index,
            y=p4(x)-2.0*stddev,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False
        ),
        go.Candlestick(x=my_index,
                       open=arr_prices.open,
                       high=arr_prices.high,
                       low=arr_prices.low,
                       close=arr_prices.close,name='Price')
        #go.Scatter(
        #    x=my_index,
        #    y=arr_prices.open,
        #    name='price',
        #    error_y=dict(
        #        type='data',
        #        symmetric=False,
        #        array=arr_prices.high-arr_prices.open,
        #        arrayminus=arr_prices.open-arr_prices.low))
    ]+smaPlots)
    
    fig.update_layout(xaxis_rangeslider_visible=False,
                      xaxis_title='Date',
                      yaxis_title=ylabelPlot,
                      title='Mean reversion %s' %outname,
                      hovermode="x")
    #cx.plot(dd, p4(xx), '-g',label='Quadratic fit')
    #plt.plot(arr_prices.high,color='red',label='High')
    #plt.plot(arr_prices.low,color='cyan',label='Low')
    #plt.plot(my_index,arr_prices.open, '+',color='orange',label='Open')

    #if len(spy_comparison)>0:  cx.set_ylabel('Price / SPY')
    #else: cx.set_ylabel('Price')
    return fig

# Fit linear regression on close
# Return the t-statistic for a given parameter estimate.
def tValLinR(close,plot=False,extra_vars=[],debug=False):
    x = np.ones((close.shape[0],2)) # adding the constant
    x[:,1] = np.arange(close.shape[0])
    if len(extra_vars)>0:
        x = np.ones((close.shape[0],3)) # adding the constant
        x[:,1] = np.arange(close.shape[0])    
        x[:,2] = extra_vars.volume#np.arange(close.shape[0])    
    ols = sm1.OLS(close, x).fit()
    if debug: print(ols.summary())
    prstd, iv_l, iv_u = wls_prediction_std(ols)
    return ols.tvalues[1] #,today_value_predicted # grab the t-statistic for the slope

# collect the slow
def slope(p4,x=[1,2]):
    j = p4(x)
    return j[1]-j[0]

# add data for new fits
def AddData(t):
    """ AddData:
          t - dataframe of minute data for  stock
    """
    t['slope']=0.0
    t['slope'] = t.close.rolling(8).apply(tValLinR)
    t['slope30']=0.0
    t['slope30'] = t.close.rolling(30).apply(tValLinR)
    t['time']  = t.index
    t['i'] = range(0,len(t))
    #t['slope_volume'] = t.volume.rolling(8).apply(tValLinR) 
    t['signif_volume'] = (t.volume-t.volume.mean())/t.volume.std()
    t['volma20'] = t.volume.rolling(20).mean()
    t['volma10'] = t.volume.rolling(10).mean()
    t['ma50'] = t.close.rolling(50).mean()
    fivma = t['ma50'].mean()
    t['ma50_div'] = (t['ma50'] - fivma)/fivma
    t['ma50_div'] *=10.0
    t['minute_diff_now'] = (t['close']-t['open'])/t['open']
    t['minute_diff_minago'] = t['minute_diff_now'].shift(1)
    t['minute_diff_2minago'] = t['minute_diff_now'].shift(2)
    t['minute_diff_15minago'] = t['minute_diff_now'].shift(15)
    t['minute_diff_5minback'] = (t['close'].shift(5)-t['open'].shift(1))/t['open'].shift(1)
    t['minute_diff_15minback'] = (t['close'].shift(15)-t['open'].shift(1))/t['open'].shift(1)
    t['minute_diff_15minfuture'] = (t['close'].shift(-15)-t['open'].shift(-1))/t['open'].shift(-1)
    t['minute_diff_5minfuture'] = (t['close'].shift(-5)-t['open'].shift(-1))/t['open'].shift(-1)
    t['minute_diff'] = (t['close']-t['open'])/t['open']
    #t['minute_diff'] = (t['close']-t['open'])/t['open']
    t['minute_diff'] *=10.0
    t['minute_diff']+=1.5
    
    t['minute_diffHL'] = (t['high']-t['low'])/t['open']
    t['minute_diffHL'] *=10.0
    t['minute_diffHL']+=1.75
    t['change'] = t.close
    t['change'] -= t.open.values[-1]
    t['change'] /= t.open.values[-1]

def generateMeanRevFigure(apiA, sqlA, tsA, tickerA, doRelativeToSpy=False,debug=False):
    """generateMeanRevFigure:
       apiA - alpaca api
       sqlA - mysql cursor
       tsA - time series api
       tickerA - str - ticker
       doRelativeToSpy - bool - compute the ratio to SPY for performance
    """
    figs=[]

    today = datetime.datetime.now(tz=est) 
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    d1_set = today.strftime("%Y-%m-%d")
    #d1_set = "2022-01-19"
    twelve_hours = (today + datetime.timedelta(hours=-12)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    minute_prices  = runTicker(apiA, tickerA, timeframe=TimeFrame.Minute, start=twelve_hours, end=d1)

    #################
    # daily prices
    #################
    daily_prices,j    = ConfigTable(tickerA, sqlA,tsA,'full',hoursdelay=18)
    spy,j    = ConfigTable('SPY', sqlA,tsA,'full',hoursdelay=18)    

    try:
        start = time.time()
        daily_prices = AddInfo(daily_prices, spy, debug=debug)
        end = time.time()
        if debug: print('Process time to add info: %s' %(end - start))
    except (ValueError,KeyError,NotImplementedError) as e:
        print("Testing multiple exceptions. {}".format(e.args[-1]))            
        print('Error processing %s' %(tickerA))
        #clean up
        print('Removing: ',tickerA)
        sqlA.cursor().execute('DROP TABLE %s' %tickerA)
    daily_prices_365d = GetTimeSlot(daily_prices,days=365)
    daily_prices_180d = GetTimeSlot(daily_prices,days=180)    
    spy_365d = GetTimeSlot(spy,days=365)
    spy_180d = GetTimeSlot(spy,days=180)    
    input_keys = ['adj_close','high','low','open','close','sma200','sma100','sma50','sma20']
    fit_365d = FitWithBand(daily_prices_365d.index,daily_prices_365d[input_keys],ticker=tickerA,outname='365d')
    fit_180d = FitWithBand(daily_prices_180d.index,daily_prices_180d[input_keys],ticker=tickerA,outname='180d')
    print(fit_180d)
    print(fit_365d)
    p_now = minute_prices['close'][-1]
    signif_180d = (p_now - fit_180d[0])/(fit_180d[1]/2)
    signif_365d = (p_now - fit_365d[0])/(fit_365d[1]/2)
    no_short = (signif_180d)>3.0 or (signif_365d)>3.0;
    no_long = (signif_180d)<-3.0 or (signif_365d)<-3.0;
    
    print('Significance dont go short: ',no_short)
    print('Significance dont go long: ',no_long)
    
    #signif_180d = 
    #sys.exit(0)
    today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (today + datetime.timedelta(days=-180)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    minute_prices_thirty = []
    minute_prices  = runTicker(apiA, tickerA, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
    minute_prices_thirty = minute_prices    
    AddData(minute_prices_thirty)

    # try mean reversion
    minute_prices_thirty['adj_close']=minute_prices_thirty['close']
    minute_prices_thirty['sma200']=minute_prices_thirty['close']
    minute_prices_thirty['sma100']=minute_prices_thirty['close']
    minute_prices_thirty['sma50']=minute_prices_thirty['close']
    input_keys = ['adj_close','high','low','open','close','sma200','sma100','sma50']
    print(minute_prices)
    prevLen=0
    m=0
    for j in range(0,500000):
    #for j in range(0,2):        
        test_minutes = -1800+j*30
        test_days = int(test_minutes/3600.0)
        minute_prices_18d = GetTimeSlot(minute_prices_thirty,18,
                                        minute_prices.index[0]+datetime.timedelta(days=test_days),
                                        addToStart=True,minutes=test_minutes)
        # checking the next 6 days to see if it gets back to the baseline
        price_after = GetTimeSlot(minute_prices_thirty,6,
                                        minute_prices.index[0]+datetime.timedelta(days=test_days+18,minutes=test_minutes+1),
                                        addToStart=True,minutes=test_minutes)
        #print(minute_prices_18d)
        #print(price_after)

        newLen=len(minute_prices_18d)
        m+=1        
        if newLen!=prevLen:
            #print(len(minute_prices_18d))
            prevLen=newLen
        else:
            continue
        if newLen==0:
            print('done')
            break
        #print(minute_prices_18d)
        fig = FitWithBand(minute_prices_18d['i'], minute_prices_18d[input_keys], ticker=tickerA,doDateKey=True, outname='60min')
        figs+=[fig]
        if m%50000==0:
            print(m)
            print(fig)
        slope_check = slope(fig[4],[fig[5],fig[5]+1])
        
        # set these slope checks using historical data?
        # at 5d or 5*500min, then 1.5 sigma. add 0.5 for each day shorter than 0.5sigma
        switch_slope = 0.00006
        signif_hi=2.0
        signif_lo=1.5
        
        if slope_check!=0:
            timeline = (fig[3]-fig[0])/slope_check
            if timeline>5*500 or timeline<0.0:
                switch_slope = slope_check
            else:
                signif_hi = 1.5+0.5*(5*500.0- timeline)/500.0
                signif_lo = 1.5+0.5*(5*500.0- timeline)/500.0
        if (slope_check>switch_slope and (fig[2])>signif_hi) or (fig[2]<-1*signif_lo and slope_check>-1*switch_slope) or (slope_check<switch_slope and (fig[2])>signif_lo) or (fig[2]<-1*signif_hi and slope_check<-1*switch_slope) :
            print('over sold or bought!',fig,minute_prices_18d.index[-1],'minute slope: %0.3f' %minute_prices_18d['slope'][-1],' p4 slope: %0.4f' %slope(fig[4],[fig[5],fig[5]+1]))
            price_after['diff_mean']=price_after['open']-fig[0]
            if len(price_after)>0:
                if price_after['diff_mean'][0]>0:
                    filter_price_after = price_after[price_after.diff_mean<0]
                    if len(filter_price_after['open'])>0:
                        p_out = filter_price_after['open'][0]
                        print('   gain of %0.3f'%(100*(fig[3]-p_out)/fig[3]),filter_price_after.index[0],' sell price: ',p_out)
                    else:
                        print('    !!!!loss')
                        #print(price_after.to_string())
                        #break
                        #sys.exit(0)
                elif price_after['diff_mean'][0]<0:
                    filter_price_after = price_after[price_after.diff_mean>0]
                    if len(filter_price_after['open'])>0:
                        p_out = filter_price_after['open'][0]
                        print('   gain of %0.3f'%(100*(p_out-fig[3])/fig[3]),filter_price_after.index[0], ' sell price: ',p_out)
                    else:
                        print('    !!!!loss')
                        #print(price_after.to_string())
                        #break
                        #sys.exit(0)
                else:
                    print ('    unclear outcome')
                    print(price_after)
        #print(minute_prices_18d[-1:])

    #fig = go.Figure(data=[go.Candlestick(x=minute_prices_thirty['i'],
    #                                         open=minute_prices_thirty['open'],
    #                                         high=minute_prices_thirty['high'],
    #                                         low=minute_prices_thirty['low'],
    #                                         close=minute_prices_thirty['close'])])
    #figs +=[ fig ]
    #print(figs)
    
    # add todays numbers if available
    if len(minute_prices)>0:
        df_today = pd.DataFrame([[minute_prices['open'][0],
                                             minute_prices['high'].max(),
                                             minute_prices['low'].min(),
                                             minute_prices['close'][-1],
                                             minute_prices['close'][-1],
                                             minute_prices['volume'].sum(),0.0,1.0]],
                                           columns=['open','high','low','close','adj_close','volume','dividendamt','splitcoef'],index=[d1_set])#index=[minute_prices.index[-1]])
        df_today.index = pd.to_datetime(df_today.index)
        if len(daily_prices)==0 or daily_prices.index[-1]<df_today.index[0]:
            daily_prices = pd.concat([daily_prices,df_today])
            daily_prices = daily_prices.sort_index()
        

    filter_shift_days = 0
    try:
        AddSMA(daily_prices)
    except ValueError:
        print('cleaning table')
    print('SMA200: ',daily_prices['sma200'][-1])
    print('SMA100: ',daily_prices['sma100'][-1])
    print('SMA50: ',daily_prices['sma50'][-1])
        
    #if filter_shift_days>0:
    #    daily_prices  = GetTimeSlot(daily_prices, days=6*365, startDate=todayFilter)
    #daily_prices_60d  = GetTimeSlot(daily_prices, days=60+filter_shift_days)
    #daily_prices_180d = GetTimeSlot(daily_prices, days=180+filter_shift_days)
    #daily_prices_365d = GetTimeSlot(daily_prices, days=365+filter_shift_days)
    #daily_prices_3y   = GetTimeSlot(daily_prices, days=3*365+filter_shift_days)
    #daily_prices_5y   = GetTimeSlot(daily_prices, days=5*365+filter_shift_days)
    #daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)
    #input_keys = ['adj_close','high','low','open','close','sma200','sma100','sma50','sma20']
    #figs+=[FitWithBand(daily_prices_60d.index, daily_prices_60d   [input_keys],ticker=tickerA,outname='60d')]
    #figs+=[FitWithBand(daily_prices_180d.index, daily_prices_180d [input_keys],ticker=tickerA,outname='180d')]
    #figs+=[FitWithBand(daily_prices_365d.index, daily_prices_365d [input_keys],ticker=tickerA,outname='365d')]
    #figs+=[FitWithBand(daily_prices_3y.index, daily_prices_3y     [input_keys],ticker=tickerA,outname='3y')]
    #figs+=[FitWithBand(daily_prices_5y.index, daily_prices_5y     [input_keys],ticker=tickerA,outname='5y')]
    return figs

#df_min = collect_latest_trades(api,df_min,False,shortThr)
ticker='WEN'
ticker='X'
#ticker='RTX'
generateMeanRevFigure(api, sqlcursor, ts, ticker, doRelativeToSpy=False)
