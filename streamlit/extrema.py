from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
import streamlit as st
import time,os
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
from matplotlib.backends.backend_agg import RendererAgg

# collect data
from ReadData import ALPACA_REST,runTicker
from alpaca_trade_api.rest import TimeFrame
import statsmodels.api as sm1
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import plotly.graph_objects as go
import plotly.figure_factory as ff

import pytz
import datetime
est = pytz.timezone('US/Eastern')
api = ALPACA_REST()

matplotlib.use("agg")
sns.set_style('darkgrid')

_lock = RendererAgg.lock

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

def collect_latest_trades(apiA,df_blah):
    """collect_latest_trades:
       apiA - alpaca api
       df_blah - dataframe with the signifances of stocks
    """
    df_blah['price_when_added']  = df_blah['current_price']
    df_blah['signif_when_added'] = df_blah['fit_diff_significance']
    df_blah['current_price'] = 0
    tickersA = df_blah['ticker'].unique()
    try:
        trade_map = apiA.get_latest_trades(tickersA)
        for t in tickersA:
            if t in trade_map:
                df_blah.loc[df_blah.ticker==t,['current_price']] = trade_map[t].p
        df_blah['fit_diff_significance'] = (df_blah['current_price'] - df_blah['fit_expectations']) / df_blah.stddev
    except:
        pass
    return df_blah

def generateFigure(apiA,tickerA):
    """collect_latest_trades:
       apiA - alpaca api
       tickerA - str - ticker
    """
    fig=None
    today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (today + datetime.timedelta(days=-13)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    minute_prices_thirty = []
    try:
        minute_prices_thirty  = runTicker(apiA, tickerA, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
        AddData(minute_prices_thirty)

        fig = go.Figure(data=go.Scatter(
            x=minute_prices_thirty['time'],
            y=minute_prices_thirty['close'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=minute_prices_thirty.high-minute_prices_thirty.close,
                arrayminus=minute_prices_thirty.close-minute_prices_thirty.low)
        ))


        
    except:
        pass

    return minute_prices_thirty,fig

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
st.write('')
row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
my_days = []
tickers_plus = []
tickers_min = []
with row2_1:
    title = st.text_input('Stock Ticker ', 'SPY')
    st.write('The current ticker is', title)

    today = datetime.datetime.now(tz=est)
    my_days = []
    for i in range(0,10):
        new_date = today+datetime.timedelta(-1*i)
        if new_date.weekday()<5:
            my_days+=[new_date]
    st.write('The current ticker is %s and number of days checked: %i' %(today,len(my_days)))
    
st.write('')
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((.1, 1, .1, 1, .1))
with row3_1, _lock:
    st.subheader('Recent highly significant')

    # Set a bar for how significant to select
    signif = st.slider('How significant to select?', 3.0, 10.0, 5.0)
    st.write("Selected ", signif, 'sigma')

    downloadFiles=False
    if st.button("Show file names"):
        st.subheader('List input files:')
        downloadFiles=True
    # Loop over recent days to find which stocks to look up
    df=[]
    for d in my_days:
        outFileName='../News/signif_%s_%s_%s.csv' %(d.day,d.month,d.year)
        if os.path.exists(outFileName):
            if downloadFiles: st.write(outFileName)
            df_part = pd.read_csv(outFileName)
            df_part['date'] = '%s-%s-%s' %(d.day,d.month,d.year)
            if len(df)==0:
                df = df_part
            else:
                df = pd.concat([df,df_part])
                
    if len(df)>0:
        st.markdown("Greater than %ssigma or overbought!" %signif)
        df = df[df.stddev>0.000001]
        df_plus = df[df.fit_diff_significance>signif]
        df_plus = collect_latest_trades(api,df_plus)
        st.table(df_plus.assign(sortkey = df_plus.groupby(['ticker'])['fit_diff_significance'].transform('max')).sort_values(['sortkey','fit_diff_significance'],ascending=[False,False]).drop('sortkey', axis=1))

        # Generating an option to draw plots
        tickers_plus = df_plus['ticker'].unique()

        # Running the oversold now
        st.write('')
        st.markdown("Greater than -%ssigma or oversold!" %signif)
        df_min = df[df.fit_diff_significance<(-1*signif)]
        df_min = collect_latest_trades(api,df_min)
        tickers_min = df_min['ticker'].unique()

        st.table(df_min.assign(sortkey = df_min.groupby(['ticker'])['fit_diff_significance'].transform('min')).sort_values(['sortkey','fit_diff_significance'],ascending=[True,True]).drop('sortkey', axis=1))

#
# now to make plots max
#
st.write('')
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((.1, 1, .1, 1, .1))
with row4_1, _lock:
    st.subheader('Select plotting - overbought')
    optionPlus = st.selectbox('Is there a stock to evaluate for overbought?',np.array(['Select']+list(tickers_plus)),key='maxTick')
    st.write('You selected:', optionPlus)

    if optionPlus!='Select':
        fig_table,fig_minute_price = generateFigure(api,optionPlus)

        # Plot!
        st.plotly_chart(fig_minute_price)
        #st.plotly_chart(fig)
        
        if st.button("Show table"):
            st.table(fig_table)
#
# now to make plots min
#
st.write('')
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns((.1, 1, .1, 1, .1))
with row5_1, _lock:
    st.subheader('Select plotting - oversold')
    optionMin = st.selectbox('Is there a stock to evaluate for oversold?',np.array(['Select']+list(tickers_min)),key='minTick')
    st.write('You selected:', optionMin)

    if optionMin!='Select':
        fig_table,fig_minute_price = generateFigure(api,optionMin)

        group_labels = ['Group 1', 'Group 2',]
        
        # Plot!
        st.plotly_chart(fig_minute_price)
        
        if st.button("Show table"):
            st.table(fig_table)


    #x1 = np.random.normal(5, 3, 5000)
    #x2 = x1-2.0
    #hist_data = [x1, x2]
    #group_labels = ['Group 1', 'Group 2',]
    #
    ## Create distplot with custom bin_size
    #fig = ff.create_distplot(
    # hist_data, group_labels, bin_size=[.1, .25, .5])
    #


        
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

