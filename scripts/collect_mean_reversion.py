import os,sys
import datetime
import base
import requests
import pandas as pd
import numpy as np
from ReadData import ALPACA_REST,runTicker,ConfigTable,ALPHA_TIMESERIES,GetTimeSlot,SQL_CURSOR
from alpaca_trade_api.rest import TimeFrame
import alpaca_trade_api
import matplotlib.dates as mdates
import pytz
import datetime,time
est = pytz.timezone('US/Eastern')
debug = True

def GetRMSData(my_index, arr_prices, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[],doRelative=False,doJoin=True, out_df = []):
    """
    my_index : datetime array
    price : array of prices or values
    ticker : str : ticker symbol name
    outname : str : name of histogram to save as
    price_key : str : name of the price to entry to fit
    spy_comparison : array : array of prices to use as a reference. don't use when None
    doRelative : bool : compute the error bands with relative changes. Bigger when there is a big change in price
    doJoin : bool : join the two arrays on matching times
    out_df : dataframe : contains the output from the fit and the significance
    """
    prices = arr_prices[price_key]
    x = mdates.date2num(my_index)
    xx = np.linspace(x.min(), x.max(), 1000)
    dd = mdates.num2date(xx)

    # prepare a spy comparison
    if len(spy_comparison)>0:
        if not doJoin:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.loc[arr_prices.index,:]
            prices /= (spy_comparison[price_key] / spy_comparison[price_key][0])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'high'] /= (spy_comparison.high / spy_comparison.high[0])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'low']  /= (spy_comparison.low  / spy_comparison.low[0])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'open'] /= (spy_comparison.open / spy_comparison.open[0])
        else:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.copy(True)
            for i in ['high','low','open','close']:
                spy_comparison[i+'_spy'] = spy_comparison[i]
                arr_prices = arr_prices.join(spy_comparison[i+'_spy'],how='left')
                if len(arr_prices[i+'_spy'])>0:
                    arr_prices[i] /= (arr_prices[i+'_spy'] / arr_prices[i+'_spy'][0])

    # perform the fit
    z4 = np.polyfit(x, prices, poly_order)
    p4 = np.poly1d(z4)

    # create an error band
    diff = prices - p4(x)
    stddev = diff.std()
    
    # relative changes
    if doRelative:
        diff /= p4(x)
        stddev = diff.std() #*p4(x).mean()

    out_arr = [p4(x)[-1],stddev,diff[-1],prices[-1]]
    output_lines = '%s,%s,%s,%s' %(p4(x)[-1],stddev,diff[-1],prices[-1])
    if stddev!=0.0:
        output_lines = '%0.3f,%0.3f,%0.3f,%s' %(p4(x)[-1],stddev,diff[-1]/stddev,prices[-1])
        out_arr = [p4(x)[-1],stddev,diff[-1]/stddev,prices[-1]]    
    if debug: print('%s,%s,%s' %(ticker,outname,output_lines))
    # check if stuff is filled
    if not (ticker in out_df['ticker'] and outname in out_df['time_span']):
        out_df = out_df.append(pd.DataFrame([[ticker,outname]+out_arr],columns=['ticker','time_span','fit_expectations','stddev','fit_diff_significance','current_price']),ignore_index=True)
    else:
        # decide what information to update! certainly the current price
        pass
if __name__ == "__main__":
    # execute only if run as a script

    # Collect APIs
    api = ALPACA_REST()
    ts = ALPHA_TIMESERIES()
    sqlcursor = SQL_CURSOR()

    # collect date and time
    filter_shift_days = 0
    today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
    outFileName='News/signif_%s_%s_%s.csv' %(today.day,today.month,today.year)
    inFileName='News/table_%s_%s_%s.csv' %(today.day-1,today.month,today.year)
    df=[]
    try:
        df = pd.read_csv(inFileName)
    except (FileNotFoundError):
        print("Testing multiple exceptions. {}".format(e.args[-1]))
        df = []
    if debug: print(df)
    todayFilter = (today + datetime.timedelta(days=-1*filter_shift_days))
    d1 = todayFilter.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (todayFilter + datetime.timedelta(days=-30)).strftime("%Y-%m-%dT%H:%M:%S-05:00")

    spy,j    = ConfigTable('SPY', sqlcursor,ts,'full',hoursdelay=18)
    if filter_shift_days>0:
        spy  = GetTimeSlot(spy, days=6*365, startDate=todayFilter)
    spy_daily_prices_60d  = GetTimeSlot(spy, days=60+filter_shift_days)
    spy_daily_prices_365d = GetTimeSlot(spy, days=365+filter_shift_days)
    spy_daily_prices_5y   = GetTimeSlot(spy, days=5*365+filter_shift_days)

    # create a dataframe to store all of the info
    df_store_data = pd.DataFrame(columns=['ticker','time_span','fit_expectations','stddev','fit_diff_significance','current_price'])
    if os.path.exists(outFileName):
        df_store_data = pd.read_csv(outFileName)
        
    #extras to process
    #base.etfs
    #base.safe_stocks
    # safe stocks to pay attention to for bigmoves
    all_tickers = []
    for safe_stock in base.safe_stocks:
        all_tickers +=[safe_stock[0]]
    for etf in base.etfs:
        all_tickers +=[etf[0]]
    while (today.hour<23 or (today.hour==23 and today.minute<30)):
        try:
            df = pd.read_csv(inFileName)
        except (FileNotFoundError):
            print("Testing multiple exceptions. {}".format(e.args[-1]))
        proc_all_tickers = all_tickers
        if len(df)>0 and 'ticker' in df.columns:
            proc_all_tickers = list(df['ticker'])+all_tickers
        for ticker in proc_all_tickers:
            if debug: print(ticker)
            # checking if it is shortable and tradeable:
            try:
                aapl_asset = api.get_asset(ticker)
                #print(aapl_asset) # this is info about it being tradeable
                #print(aapl_asset.shortable)
            except (alpaca_trade_api.rest.APIError,requests.exceptions.HTTPError):
                print("Testing multiple exceptions. {}".format(e.args[-1]))
                continue
            hour_prices_thirty=[]
            minute_prices_thirty=[]
            hour_prices_thirty_spy=[]
            minute_prices_thirty_spy=[]
            while len(hour_prices_thirty)==0 and len(minute_prices_thirty)==0 and len(hour_prices_thirty_spy)==0 and len(minute_prices_thirty_spy)==0:
                try:
                    hour_prices_thirty    = runTicker(api, ticker, timeframe=TimeFrame.Hour, start=thirty_days, end=d1)
                    minute_prices_thirty  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
                    hour_prices_thirty_spy    = runTicker(api, 'SPY', timeframe=TimeFrame.Hour, start=thirty_days, end=d1)
                    minute_prices_thirty_spy  = runTicker(api, 'SPY', timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
                except (alpaca_trade_api.rest.APIError):
                    print("Testing multiple exceptions. {}".format(e.args[-1]))
                continue
            hour_prices_10d       = GetTimeSlot(hour_prices_thirty,      days=10)
            minute_prices_10d     = GetTimeSlot(minute_prices_thirty,    days=10)
            hour_prices_spy_10d   = GetTimeSlot(hour_prices_thirty_spy,  days=10)
            minute_prices_spy_10d = GetTimeSlot(minute_prices_thirty_spy,days=10)
    
            daily_prices,j    = ConfigTable(ticker, sqlcursor,ts,'full',hoursdelay=18)
            if filter_shift_days>0:
                daily_prices  = GetTimeSlot(daily_prices, days=6*365, startDate=todayFilter)
            daily_prices_60d  = GetTimeSlot(daily_prices, days=60+filter_shift_days)
            daily_prices_180d = GetTimeSlot(daily_prices, days=180+filter_shift_days)
            daily_prices_365d = GetTimeSlot(daily_prices, days=365+filter_shift_days)
            daily_prices_3y   = GetTimeSlot(daily_prices, days=3*365+filter_shift_days)
            daily_prices_5y   = GetTimeSlot(daily_prices, days=5*365+filter_shift_days)
            daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)
            #print('ticker,time_span,fit_difference_in_stddev,current_price,stddev')
            if debug: print('ticker,time_span,fit_expectations,stddev,fit_diff_significance,current_price')
    
            # Run:
            GetRMSData(daily_prices_60d.index, daily_prices_60d [['adj_close','high','low','open','close']],ticker=ticker,outname='60d',out_df = df_store_data)
            GetRMSData(daily_prices_180d.index,daily_prices_180d[['adj_close','high','low','open','close']],ticker=ticker,outname='180d',out_df = df_store_data)
            GetRMSData(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],ticker=ticker,outname='365d',out_df = df_store_data)
            GetRMSData(daily_prices_3y.index,  daily_prices_3y  [['adj_close','high','low','open','close']],ticker=ticker,outname='3y',out_df = df_store_data)
            GetRMSData(daily_prices_5y.index,  daily_prices_5y  [['adj_close','high','low','open','close']],ticker=ticker,outname='5y', doRelative=False,out_df = df_store_data)
            #GetRMSData(my_index, arr_prices, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[],doJoin=True)
            GetRMSData(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],
                    ticker=ticker,outname='365dspycomparison',spy_comparison = spy_daily_prices_365d[['adj_close','high','low','open','close']],out_df = df_store_data)
            GetRMSData(daily_prices_60d.index,daily_prices_60d[['adj_close','high','low','open','close']],
                    ticker=ticker,outname='60dspycomparison',spy_comparison = spy_daily_prices_60d[['adj_close','high','low','open','close']],out_df = df_store_data)
            GetRMSData(daily_prices_5y.index,daily_prices_5y[['adj_close','high','low','open','close']],
                    ticker=ticker,outname='5yspycomparison',spy_comparison = spy_daily_prices_5y[['adj_close','high','low','open','close']],out_df = df_store_data)
    
            # Spy comparison as well as mintue 10day comparison for minute and hour data
            GetRMSData(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                    ticker=ticker,outname='10dhspycomparison', price_key='close',spy_comparison = hour_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True,out_df = df_store_data)
            GetRMSData(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                    ticker=ticker,outname='10dh', price_key='close',out_df = df_store_data)
            GetRMSData(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                    ticker=ticker,outname='10dmspycomparison', price_key='close',spy_comparison = minute_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True,out_df = df_store_data)
            GetRMSData(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                ticker=ticker,outname='10dm', price_key='close',out_df = df_store_data)
        
        # Stalling before the next cycle
        df_store_data.to_csv(outFileName,index=False)
        time.sleep(60)
        
