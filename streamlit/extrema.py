from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
import streamlit as st
import time,os
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
from ReadData import ALPACA_REST,runTicker,ALPHA_TIMESERIES,SQL_CURSOR,ConfigTable,GetTimeSlot,ALPHA_FundamentalData
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

_lock = RendererAgg.lock

def clear_form():
    st.session_state["tickerKey"] = "Select"

def FitWithBand(my_index, arr_prices, doMarker=True, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[], doRelative=False, doJoin=True, debug=False):
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
    x = mdates.date2num(my_index)
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
    z4 = np.polyfit(x, prices, poly_order)
    p4 = np.poly1d(z4)

    # create an error band
    diff = prices - p4(x)
    stddev = diff.std()

    output_lines = '%s,%s,%s,%s' %(p4(x)[-1],stddev,diff[-1],prices[-1])
    if stddev!=0.0:
        output_lines = '%0.3f,%0.3f,%0.3f,%s' %(p4(x)[-1],stddev,diff[-1]/stddev,prices[-1])    
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
    if doRelative:
        stddev *= p4(x).mean()

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
            line=dict(color='rgb(31, 119, 180)'),
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
    ])
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

# add data for new fits
def AddData(t):
    """ AddData:
          t - dataframe of minute data for  stock
    """
    t['slope']=0.0
    t['slope'] = t.close.rolling(8).apply(tValLinR)
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

def collect_latest_trades(apiA,df_blah,getShortable=True,shortThr=5.0):
    """collect_latest_trades:
       apiA - alpaca api
       df_blah - dataframe with the signifances of stocks
    """
    df_blah['price_when_added']  = df_blah['current_price']
    df_blah.loc[:,'signif_when_added'] = df_blah['fit_diff_significance']
    df_blah['current_price'] = 0
    #df_blah['fit_diff_significance'] = 0
    df_blah['shortable'] = False
    tickersA = df_blah['ticker'].unique()
    print('Processing: %s' %len(tickersA))
    #sys.stdout.flush()
    try:
        trade_map = apiA.get_latest_trades(tickersA)
        print('have trades')
        for t in tickersA:
            asset_info=None
            if t in trade_map:
                df_blah.loc[df_blah.ticker==t,['current_price']] = trade_map[t].p
                df_blah_stock = df_blah[df_blah.ticker==t].copy(True)
                df_blah_stock['signfi_tmp'] = (df_blah_stock['current_price']-df_blah_stock['fit_expectations'])/df_blah_stock.stddev
                signif_tmp = df_blah_stock['signfi_tmp'].max()
            df_blah.loc[df_blah.ticker==t,['shortable']] = False
            df_blah.loc[df_blah.ticker==t,['shortable_checked']] = 'no'
            if getShortable and signif_tmp>shortThr:
                try:
                    asset_info = apiA.get_asset(t)
                except alpaca_trade_api.rest.APIError:
                    print('Could not find %s' %t)
                    continue
                
                if t in trade_map and getShortable:
                    df_blah.loc[df_blah.ticker==t,['shortable']] = asset_info.shortable
                    df_blah.loc[df_blah.ticker==t,['shortable_checked']] = 'yes'
            elif signif_tmp>shortThr:
                df_blah.loc[df_blah.ticker==t,['shortable']] = False
                
        df_blah['fit_diff_significance'] = (df_blah['current_price'] - df_blah['fit_expectations']) / df_blah.stddev
    except Exception as e:
        print(e)
        pass
    print('return')
    return df_blah

def generateMeanRevFigure(apiA,sqlA,tsA,tickerA,doRelativeToSpy=False):
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

    daily_prices,j    = ConfigTable(tickerA, sqlA,tsA,'full',hoursdelay=18)

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
        
    #st.table(daily_prices.tail())
    filter_shift_days = 0
    if filter_shift_days>0:
        daily_prices  = GetTimeSlot(daily_prices, days=6*365, startDate=todayFilter)
    daily_prices_60d  = GetTimeSlot(daily_prices, days=60+filter_shift_days)
    daily_prices_180d = GetTimeSlot(daily_prices, days=180+filter_shift_days)
    daily_prices_365d = GetTimeSlot(daily_prices, days=365+filter_shift_days)
    daily_prices_3y   = GetTimeSlot(daily_prices, days=3*365+filter_shift_days)
    daily_prices_5y   = GetTimeSlot(daily_prices, days=5*365+filter_shift_days)
    daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)

    figs+=[FitWithBand(daily_prices_60d.index, daily_prices_60d [['adj_close','high','low','open','close']],ticker=tickerA,outname='60d')]
    figs+=[FitWithBand(daily_prices_180d.index, daily_prices_180d [['adj_close','high','low','open','close']],ticker=tickerA,outname='180d')]
    figs+=[FitWithBand(daily_prices_365d.index, daily_prices_365d [['adj_close','high','low','open','close']],ticker=tickerA,outname='365d')]
    figs+=[FitWithBand(daily_prices_3y.index, daily_prices_3y [['adj_close','high','low','open','close']],ticker=tickerA,outname='3y')]
    figs+=[FitWithBand(daily_prices_5y.index, daily_prices_5y [['adj_close','high','low','open','close']],ticker=tickerA,outname='5y')]

    # Compute relative to spy
    if doRelativeToSpy:
        spy,j    = ConfigTable('SPY', sqlA,tsA,'full',hoursdelay=18)
        if filter_shift_days>0:
            spy  = GetTimeSlot(spy, days=6*365, startDate=todayFilter)
        spy_daily_prices_60d  = GetTimeSlot(spy, days=60+filter_shift_days)
        spy_daily_prices_365d = GetTimeSlot(spy, days=365+filter_shift_days)
        spy_daily_prices_5y   = GetTimeSlot(spy, days=5*365+filter_shift_days)

        figs+=[FitWithBand(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],
                           ticker=tickerA,outname='365dspycomparison',spy_comparison = spy_daily_prices_365d[['adj_close','high','low','open','close']])]
        figs+=[FitWithBand(daily_prices_60d.index,daily_prices_60d[['adj_close','high','low','open','close']],
                           ticker=tickerA,outname='60dspycomparison',spy_comparison = spy_daily_prices_60d[['adj_close','high','low','open','close']])]
        figs+=[FitWithBand(daily_prices_5y.index,daily_prices_5y[['adj_close','high','low','open','close']],
                           ticker=tickerA,outname='5yspycomparison',spy_comparison = spy_daily_prices_5y[['adj_close','high','low','open','close']])] 
    return figs

def generateFigure(apiA,tickerA):
    """generateFigure:
       apiA - alpaca api
       tickerA - str - ticker
    """
    figs=[]
    doSPYComparison=False
    today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (today + datetime.timedelta(days=-13)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    minute_prices_thirty = []
    try:
        minute_prices_thirty  = runTicker(apiA, tickerA, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
        AddData(minute_prices_thirty)
        minute_prices_spy=[]
        if doSPYComparison:
            minute_prices_spy  = runTicker(apiA, 'SPY', timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
            AddData(minute_prices_spy)
            minute_prices_spy['signif_volume_spy'] = minute_prices_spy.signif_volume
            minute_prices_spy['change_spy'] = minute_prices_spy.change
            minute_prices_spy['minute_diffHL_spy'] = minute_prices_spy.minute_diffHL
            minute_prices_spy['minute_diff_spy'] = minute_prices_spy.minute_diff
            
            minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.signif_volume_spy,how='left')
            minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.change_spy,how='left')
            minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.minute_diff_spy,how='left')
            minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.minute_diffHL_spy,how='left')
            minute_prices_thirty['signif_volume_over_spy'] = minute_prices_thirty.signif_volume / minute_prices_thirty.signif_volume_spy
            minute_prices_thirty['change_over_spy'] = minute_prices_thirty.change - minute_prices_thirty.change_spy
            minute_prices_thirty['signif_volume_over_spy'].where(minute_prices_thirty['signif_volume_over_spy']<150.0,150.0,inplace=True)
            minute_prices_thirty['signif_volume_over_spy'].where(minute_prices_thirty['signif_volume_over_spy']>-150.0,-150.0,inplace=True)
        # draw figures
        fig = go.Figure(data=go.Scatter(
            x=minute_prices_thirty['i'],
            y=minute_prices_thirty['close'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=minute_prices_thirty.high-minute_prices_thirty.close,
                arrayminus=minute_prices_thirty.close-minute_prices_thirty.low)
        ))
        fig.update_layout(xaxis_title="Minute Bars",yaxis_title="%s Price" %tickerA)
        figs +=[ fig ]

        fig = go.Figure(data=[go.Candlestick(x=minute_prices_thirty['i'],
                                             open=minute_prices_thirty['open'],
                                             high=minute_prices_thirty['high'],
                                             low=minute_prices_thirty['low'],
                                             close=minute_prices_thirty['close'])])
        figs +=[ fig ]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(  go.Scatter(x=minute_prices_thirty['i'],
                                   y=minute_prices_thirty['close'],name='price'),
            secondary_y=False)
        if doSPYComparison:
            fig.add_trace(  go.Scatter(x=minute_prices_thirty['i'],
                                       y=minute_prices_thirty['signif_volume_over_spy'],name='volume sigificance/SPY'),secondary_y=True)
                            
        
        fig.add_trace(  go.Scatter(x=minute_prices_thirty['i'],
                                   y=minute_prices_thirty['slope'],name='slope'),
                        secondary_y=True)
        fig.add_trace(  go.Scatter(x=minute_prices_thirty['i'],
                                   y=minute_prices_thirty['signif_volume'],name='volume sigificance'),
                        secondary_y=True)

        # Set x-axis title
        fig.update_xaxes(title_text="Minute Bars")

        # Set y-axes titles
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Slope", secondary_y=True)
        figs+=[fig]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=minute_prices_thirty['time'],
            y=minute_prices_thirty['close'],
            name='price',
            error_y=dict(
                type='data',
                symmetric=False,
                array=minute_prices_thirty.high-minute_prices_thirty.close,
                arrayminus=minute_prices_thirty.close-minute_prices_thirty.low)),
            secondary_y=False)
        fig.add_trace(  go.Scatter(x=minute_prices_thirty['time'],
                                   y=minute_prices_thirty['slope'],name='slope'),
                        secondary_y=True)
        fig.add_trace(  go.Scatter(x=minute_prices_thirty['time'],
                                   y=minute_prices_thirty['signif_volume'],name='volume sigificance'),
                        secondary_y=True)

        # Set x-axis title
        fig.update_layout(xaxis_title="Date",yaxis_title="%s Price" %tickerA)

        # Set y-axes titles
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Slope", secondary_y=True)

        figs +=[ fig ]
        
    except:
        pass

    return minute_prices_thirty,figs

def generateFigurePanda(ar,tickerA,xaxis=[],yaxis1=[],yaxis2=[],ytitle='EPS'):
    """generateFigurePanda
       ar - panda array
       tickerA - str - ticker
    """
    figs=[]
    if len(ar)==0:
        return figs

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for y in yaxis1:
        fig.add_trace(go.Scatter(
            x=ar[xaxis],
            y=ar[y],
            name=y),secondary_y=False)
    for y in yaxis2:        
        fig.add_trace(  go.Scatter(x=ar[xaxis],
                                   y=ar[y],name=y),
                        secondary_y=True)

    # Set x-axis title
    fig.update_layout(xaxis_title="Date",yaxis_title="%s %s" %(tickerA,ytitle))
    
    # Set y-axes titles
    #fig.update_yaxes(title_text="Price", secondary_y=False)
    if len(yaxis2)>0:
        fig.update_yaxes(title_text=yaxis2[0], secondary_y=True)

    #fig.write_image("fig1.png")
    figs +=[ fig ]
    return figs

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

# Generating Tickers
row0_1.title('Monitoring stocks hitting extreme values')
row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))
with row1_1:
    st.markdown("This looks at stocks on finviz and our hopefully more stable list to look for excesses. We probe the last 10 days for all that exceeded limits to search for very significant excesses.")

    loadEarnings = st.checkbox('Load upcoming earnings',key='loadEarnings')
    if loadEarnings:
        today = datetime.datetime.now(tz=est)
        future = datetime.datetime.now(tz=est)+datetime.timedelta(days=5)
        stock_earnings = pd.read_csv(STOCK_DB_PATH+'/stockEarnings.csv')
        stock_earnings['reportDate']=pd.to_datetime(stock_earnings['reportDate'])
        stock_earnings = stock_earnings[(stock_earnings.reportDate < '%s-%s-%s' %(future.year,future.month,future.day)) & (stock_earnings.reportDate >= '%s-%s-%s' %(today.year,today.month,today.day)) ]
        st.dataframe(stock_earnings)
        
st.write('')
row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
my_days = []
tickers_plus = []
tickers_min = []
with row2_1:

    titleStart='Select'
    st.button("Clear",on_click=clear_form)
    title = st.text_input('Stock Ticker ', titleStart,key='tickerKey')
    title = title.upper()
    st.write('The current ticker is', title)

    today = datetime.datetime.now(tz=est)
    my_days = []
    for i in range(0,10):
        new_date = today+datetime.timedelta(-1*i)
        if new_date.weekday()<5:
            my_days+=[new_date]
    st.write('The current ticker is %s and number of days checked: %i' %(today,len(my_days)))
    #st.table(pd.DataFrame(list(api.get_asset(title)._raw.items())))

    doRelativeToSpyAll = st.checkbox('Show relative to SPY',key='relTitleSpy')
    doEarnings = st.checkbox('Show earnings',key='do earnings')
    
    if title!='SELECT':
        # Print a table of stock information on Alpaca
        st.json(api.get_asset(title)._raw)

        # Collect figures
        fig_table,figs_minute_price = generateFigure(api,title)

        # Plot!
        for fig_minute_price in figs_minute_price:
            st.plotly_chart(fig_minute_price)

        mean_figs = generateMeanRevFigure(api,sqlcursor,ts,title,doRelativeToSpyAll)
        st.markdown('Number of entries: %s' %len(mean_figs))
        for mean_fig in mean_figs:
            st.plotly_chart(mean_fig)
        
        if st.button("Show table"):
            st.dataframe(fig_table)

        if doEarnings:
            st.markdown("**Earnings info**")
            fd=ALPHA_FundamentalData()
            sqlcursorShorts = SQL_CURSOR(db_name=STOCK_DB_PATH+'/stocksShort.db')
            
            sqlcursorCal = SQL_CURSOR(db_name=STOCK_DB_PATH+'/earningsCalendarv2.db')
            fiscal_figs = []
            ticker = title
            annualE,qE = GetPastEarnings(fd,ticker,sqlcursorCal)
            overview = GetStockOverview(fd,ticker,sqlcursorCal) #fd.get_company_overview(ticker) # has P/E, etc
            if len(overview)>0:
                fiscal_figs+=generateFigurePanda(overview,ticker,xaxis='Date',yaxis1=['BookValue'],yaxis2=[],ytitle='BookValue')
                fiscal_figs+=generateFigurePanda(overview,ticker,xaxis='Date',yaxis1=['SharesShort','SharesShortPriorMonth'],yaxis2=['ShortPercentFloat'],ytitle='Shares Short')                

            if len(qE)>0:
                fiscal_figs+=generateFigurePanda(qE,ticker,xaxis='reportedDate',yaxis1=['reportedEPS','estimatedEPS'],yaxis2=['surprise'])
            if len(annualE)>0:
                fiscal_figs+=generateFigurePanda(annualE,ticker,xaxis='fiscalDateEnding',yaxis1=['reportedEPS'],yaxis2=[])
                
            incomeQ = GetIncomeStatement(fd, ticker, annual=False, debug=False)
            incomeA = GetIncomeStatement(fd, ticker, annual=True, debug=False)

            if len(incomeQ)>0:
                fiscal_figs+=generateFigurePanda(incomeQ,ticker,xaxis='fiscalDateEnding',yaxis1=['totalRevenue','costOfRevenue'],yaxis2=['netIncome'],ytitle='New Income')
            if len(incomeA)>0:
                fiscal_figs+=generateFigurePanda(incomeA,ticker,xaxis='fiscalDateEnding',yaxis1=['totalRevenue','costOfRevenue'],yaxis2=['netIncome'],ytitle='New Income')
    
            balanceQ = GetBalanceSheetQuarterly(fd, ticker, debug=False)
            #balanceA = GetBalanceSheetAnnual(fd, ticker, debug=False)
            if 'currentDebt' in balanceQ:
                balanceQ['debt_to_assets'] = balanceQ['currentDebt'] / balanceQ['totalCurrentAssets']
                balanceQ['debt_to_commonstock'] = balanceQ['currentDebt'] / balanceQ['commonStock']
                fiscal_figs+=generateFigurePanda(balanceQ,ticker,xaxis='fiscalDateEnding',yaxis1=['debt_to_assets','debt_to_commonstock'],yaxis2=['commonStock'],ytitle='Common Stock')
                fiscal_figs+=generateFigurePanda(balanceQ,ticker,xaxis='fiscalDateEnding',yaxis1=['currentDebt','shortTermDebt','currentLongTermDebt','cashAndShortTermInvestments'],yaxis2=['commonStock'],ytitle='Common Stock')
            
            for fiscal_fig in fiscal_figs:
                st.plotly_chart(fiscal_fig)

st.write('')
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((.1, 1, .1, 1, .1))
do_refresh_table=False
with row3_1, _lock:
    st.subheader('Recent highly significant')

    # Set a bar for how significant to select
    signif = st.slider('How significant to select?', 3.0, 10.0, 5.0)
    st.write("Selected ", signif, 'sigma')
    shortThr = st.slider('Short significance threshold to check (slow to check, so set the threshold high to avoid long waiting)?', 0.0, 10.0, 5.0)
    st.write("Selected ", shortThr, 'short threshold')
    signif_start = st.slider('How significant to select for added stocks?', 1.0, 10.0, 4.0,key='startSig')
    st.write("Selected ", signif_start, 'sigma')
    signif_start_etf = st.slider('How significant to select for ETFs?', 0.0, 10.0, 2.0,key='startSigETF')
    st.write("Selected ", signif_start_etf, 'sigma')    

    downloadFiles=False
    if st.button("Show file names"):
        st.subheader('List input files:')
        downloadFiles=True
        
    # Loop over recent days to find which stocks to look up
    df=[]
    fsplits =[]
    if os.path.exists('%s/split_stocks.txt' %STOCK_DB_PATH):
        fsplit = open('%s/split_stocks.txt' %STOCK_DB_PATH)
        for s1 in fsplit: fsplits+=[s1.rstrip('\n').strip()]
    for d in my_days:
        outFileName='%s/News/signif_%s_%s_%s.csv' %(STOCK_DB_PATH,d.day,d.month,d.year)
        if os.path.exists(outFileName):
            if downloadFiles: st.write(outFileName)
            df_part = pd.read_csv(outFileName)
            if len(df_part[df_part.ticker.isin(fsplits)])>0:
                df_part[~df_part.ticker.isin(fsplits)].to_csv(outFileName,index=False)
                if os.path.exists('%s/split_stocks.txt' %STOCK_DB_PATH):
                    os.remove('%s/split_stocks.txt' %STOCK_DB_PATH)
            df_part['date'] = '%s-%s-%s' %(d.day,d.month,d.year)
            if len(df)==0:
                df = df_part[~df_part.ticker.isin(fsplits)]
            else:
                df = pd.concat([df,df_part[~df_part.ticker.isin(fsplits)]])


    if len(df)>0:
        do_refresh_table = st.checkbox('Refresh table?')
        do_static_table = st.checkbox('Draw static table')
        add_10dm = st.checkbox('Add 10 day by minute checks')
        add_10dh = st.checkbox('Add 10 day by hour checks')
        require_shortable = st.checkbox('Require shortable')
        rm_spycomparison = st.checkbox('Remove SPY comparison')
        optionTFrame = st.selectbox('Select a time frame?',np.array(['Select','60d','365d','180d','3y','5y']),key='tframe')
        keys_for_list = list(df['time_span'].unique())
        if not add_10dm:
            keys_for_list.remove('10dm')
            keys_for_list.remove('10dmspycomparison')
        if not add_10dh:
            keys_for_list.remove('10dh')
            keys_for_list.remove('10dhspycomparison')        
        df = df[df['time_span'].isin(keys_for_list)]
                                     
        st.markdown("Greater than %ssigma or overbought!" %signif)
        df = df[df.stddev>0.000001]

        df_plus = df[df.fit_diff_significance>signif_start].copy(True)
        if rm_spycomparison:
            df_plus  = df_plus[~df_plus['time_span'].str.lower().str.contains('comparison')]
        if optionTFrame!='Select':
            df_plus  = df_plus[df_plus['time_span'].str.lower().str.contains(optionTFrame)]
        if do_refresh_table:
            df_plus = collect_latest_trades(api,df_plus,True,shortThr)
        df_plus = df_plus[df_plus.fit_diff_significance>signif]  
        if require_shortable:
            df_plus = df_plus[df_plus.shortable]
        df_plus = df_plus.assign(sortkey = df_plus.groupby(['ticker'])['fit_diff_significance'].transform('max')).sort_values(['sortkey','fit_diff_significance'],ascending=[False,False]).drop('sortkey', axis=1)
        if do_static_table:
            #df_plus.style.hide_index()
            try:
                st.table(df_plus.style.highlight_max(axis=0))
            except:
                st.table(df_plus)
        else:
            st.dataframe(df_plus,500,500)

        # Generating an option to draw plots
        tickers_plus = df_plus['ticker'].unique()

        # Running the oversold now
        st.write('')
        st.markdown("Greater than -%ssigma or oversold!" %signif)
        #df_min = df.copy(True)
        df_min = df[df.fit_diff_significance<(-1*signif_start)].copy(True)
        if rm_spycomparison:
            df_min  = df_min[~df_min['time_span'].str.lower().str.contains('comparison')]
        if optionTFrame!='Select':
            df_min  = df_min[df_min['time_span'].str.lower().str.contains(optionTFrame)]
        if do_refresh_table:
            df_min = collect_latest_trades(api,df_min,False,shortThr)
        df_min = df_min[df_min.fit_diff_significance<(-1*signif)]
        if require_shortable and shortThr<3.0:
            df_min = df_min[df_min.shortable]

        tickers_min = df_min['ticker'].unique()

        df_min = df_min.assign(sortkey = df_min.groupby(['ticker'])['fit_diff_significance'].transform('min')).sort_values(['sortkey','fit_diff_significance'],ascending=[True,True]).drop('sortkey', axis=1)
        if do_static_table:
            #df_min.style.hide_index()
            try:
                st.table(df_min.style.highlight_max(axis=0))
            except:
                st.table(df_min)
        else:
            st.dataframe(df_min,500,500)

        # Running the stable etfs and other stable stocks. a gut feeling about stability, but not a real measure
        st.write('')
        st.markdown("ETFs and stable versions!")
        # base.safe_stocks, base.etfs
        keep_list = []
        for j in base.safe_stocks: keep_list +=[j[0]]
        for j in base.etfs: keep_list +=[j[0]]
        df_etf = df[df['ticker'].isin(keep_list)].copy(True)
        # only keep the most recent because we are drawing all of them
        df_etf['date'] = pd.to_datetime(df_etf['date'])
        most_recent_date = df_etf['date'].max()
        df_etf = df_etf[df_etf.date == most_recent_date ]
        if rm_spycomparison:
            df_etf  = df_etf[~df_etf['time_span'].str.lower().str.contains('comparison')]
        if optionTFrame!='Select':
            df_etf  = df_etf[df_etf['time_span'].str.lower().str.contains(optionTFrame)]

        if do_refresh_table:
            df_etf = collect_latest_trades(api,df_etf,False,shortThr)

        df_etf = df_etf[abs(df_etf.fit_diff_significance)>signif_start_etf]
        
        if require_shortable and shortThr<3.0:
            df_etf = df_etf[df_etf.shortable]
        tickers_etf = df_etf['ticker'].unique()
        df_etf['abs_sort'] = df_etf['fit_diff_significance'].abs()
        df_etf = df_etf.assign(sortkey = df_etf.groupby(['ticker'])['abs_sort'].transform('max')).sort_values(['sortkey','abs_sort'],ascending=[False,False]).drop('sortkey', axis=1).drop('abs_sort', axis=1)
        if do_static_table:
            try:
                st.table(df_etf.style.highlight_max(axis=0))
            except:
                st.table(df_etf)
        else:
            st.dataframe(df_etf,500,500)
        
#
# now to make plots max
#
st.write('')
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((.1, 1, .1, 1, .1))
with row4_1, _lock:
    st.subheader('Select plotting - overbought')
    doRelativeToSpy = st.checkbox('Show relative to SPY',key='maxRelSpyBought')
    optionPlus = st.selectbox('Is there a stock to evaluate for overbought?',np.array(['Select']+list(tickers_plus)),key='maxTick')
    st.write('You selected:', optionPlus)

    if optionPlus!='Select':
        fig_table,figs_minute_price = generateFigure(api,optionPlus)

        # Print a table of stock information on Alpaca
        st.json(api.get_asset(optionPlus)._raw)
        
        # Plot!
        for fig_minute_price in figs_minute_price:
            st.plotly_chart(fig_minute_price)
        #st.plotly_chart(fig)

        mean_figs = generateMeanRevFigure(api,sqlcursor,ts,optionPlus,doRelativeToSpy)
        st.markdown('Number of entries: %s' %len(mean_figs))
        for mean_fig in mean_figs:
            st.plotly_chart(mean_fig)
        
        if st.button("Show table"):
            st.dataframe(fig_table)
#
# now to make plots min
#
st.write('')
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns((.1, 1, .1, 1, .1))
with row5_1, _lock:
    st.subheader('Select plotting - oversold')
    doRelativeToSpy = st.checkbox('Show relative to SPY',key='minRelSpySold')
    optionMin = st.selectbox('Is there a stock to evaluate for oversold?',np.array(['Select']+list(tickers_min)),key='minTick')
    st.write('You selected:', optionMin)

    if optionMin!='Select':
        fig_table,figs_minute_price = generateFigure(api,optionMin)

        # Print a table of stock information on Alpaca
        st.json(api.get_asset(optionMin)._raw)

        # Plot!
        for fig_minute_price in figs_minute_price:
            st.plotly_chart(fig_minute_price)

        mean_figs = generateMeanRevFigure(api,sqlcursor,ts,optionMin,doRelativeToSpy)
        for mean_fig in mean_figs:
            st.plotly_chart(mean_fig)

            
        if st.button("Show table"):
            st.dataframe(fig_table)

#
# now load the earnings lists and other interesting news stories
#
st.write('')
row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns((.1, 1, .1, 1, .1))
with row6_1, _lock:
    st.subheader('News analysis')
    showNewsTable = st.checkbox('Show tables',key='newTables')
    doRelativeToSpy = st.checkbox('Show relative to SPY',key='minRelSpyNews')
    for istory in [STOCK_DB_PATH+'/Instructions/out_earnings_instructions.csv',
                   STOCK_DB_PATH+'/Instructions/out_bull_instructions.csv',
                   STOCK_DB_PATH+'/Instructions/out_target_instructions.csv',
                   STOCK_DB_PATH+'/Instructions/out_pharmaphase_instructions.csv',
                   STOCK_DB_PATH+'/Instructions/out_upgrade_instructions.csv',]:
        if os.path.exists(istory):
            news_type = istory[istory.find('out_')+4:-len('_instructions.csv')]
            st.markdown('**%s**' %news_type)
            itable = pd.read_csv(istory,sep=' ')
            if showNewsTable:
                 st.table(itable)
            news_tickers = []
            if 'ticker' in itable.columns:
                 news_tickers = itable['ticker'].unique()

            optionNews = st.selectbox('Select Ticker',np.array(['Select']+list(news_tickers)),key='select_%s' %news_type)
            st.write('You selected:', optionNews)

            if optionNews!='Select':
                 fig_table,figs_minute_price = generateFigure(api,optionNews)

                 # Print a table of stock information on Alpaca
                 st.json(api.get_asset(optionNews)._raw)
        
                 # Plot!
                 for fig_minute_price in figs_minute_price:
                     st.plotly_chart(fig_minute_price)

                 mean_figs = generateMeanRevFigure(api,sqlcursor,ts,optionNews,doRelativeToSpy)
                 st.markdown('Number of entries: %s' %len(mean_figs))
                 for mean_fig in mean_figs:
                     st.plotly_chart(mean_fig)
        
if st.button("Download"):
    with row3_1, _lock:
        st.subheader('Generate a random plot')

        fig = Figure()
        ax = fig.subplots()
        sns.histplot(np.random.normal(5, 3, 5000),binrange=(0,20.0),ax=ax)
        ax.set_xlabel('Normal Dist')
        ax.set_ylabel('Density')
        st.pyplot(fig)
        st.markdown("It looks like you've read a gr")

