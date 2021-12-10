import os,sys,requests
import urllib3
import datetime
import base
import pandas as pd
import numpy as np
from ReadData import ALPACA_REST,runTicker,ConfigTable,ALPHA_TIMESERIES,GetTimeSlot,SQL_CURSOR
from alpaca_trade_api.rest import TimeFrame
import alpaca_trade_api
import matplotlib.dates as mdates
import pytz
import datetime,time
est = pytz.timezone('US/Eastern')
debug = False

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
            for i in ['high','low','open','close',price_key]:
                spy_comparison[i+'_spy'] = spy_comparison[i]
                arr_prices = arr_prices.join(spy_comparison[i+'_spy'],how='left')
                if len(arr_prices[i+'_spy'])>0:
                    arr_prices[i] /= (arr_prices[i+'_spy'] / arr_prices[i+'_spy'][0])
            prices = arr_prices[price_key]

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
    if len(out_df[(out_df['ticker']==ticker) & (out_df['time_span']==outname)])==0:
        out_df = out_df.append(pd.DataFrame([[ticker,outname]+out_arr],columns=['ticker','time_span','fit_expectations','stddev','fit_diff_significance','current_price']),ignore_index=True)
    else:
        # decide what information to update! certainly the current price
        pass
    return out_df

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
    inFileName='News/table_%s_%s_%s.csv' %(today.day,today.month,today.year) # HACK!!!
    df=[]
    try:
        df = pd.read_csv(inFileName)
    except (FileNotFoundError) as e:
        print("Testing multiple exceptions. {}".format(e.args[-1]))
        df = []
    if debug: print(df)
    todayFilter = (today + datetime.timedelta(days=-1*filter_shift_days))
    d1 = todayFilter.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (todayFilter + datetime.timedelta(days=-30)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    one_day = (todayFilter + datetime.timedelta(days=-1)).strftime("%Y-%m-%dT%H:%M:%S-05:00")

    spy,j    = ConfigTable('SPY', sqlcursor,ts,'full',hoursdelay=18)
    if filter_shift_days>0:
        spy  = GetTimeSlot(spy, days=6*365, startDate=todayFilter)
    spy_daily_prices_60d  = GetTimeSlot(spy, days=60+filter_shift_days)
    spy_daily_prices_365d = GetTimeSlot(spy, days=365+filter_shift_days)
    spy_daily_prices_5y   = GetTimeSlot(spy, days=5*365+filter_shift_days)

    # create a dataframe to store all of the info
    curr_results = pd.DataFrame(columns=['ticker','quote','trade','bar_close','bar_high','bar_low','bar_open'])
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
        except (FileNotFoundError) as e:
            print("Testing multiple exceptions. {}".format(e.args[-1]))

        # check the quotes:
        if len(df_store_data)>0:
            sns={}
            try:
                sns = api.get_snapshots(list(df_store_data['ticker'].unique()))
            except (alpaca_trade_api.rest.APIError,requests.exceptions.HTTPError,ValueError,urllib3.exceptions.ProtocolError,ConnectionResetError,urllib3.exceptions.ProtocolError,ConnectionResetError,requests.exceptions.ConnectionError) as e:
                    print("Testing multiple exceptions for snapshots. {}".format(e.args[-1]))
                    continue
            curr_results = pd.DataFrame(columns=['ticker','quote_ap','quote_bp','trade','bar_close','bar_high','bar_low','bar_open'])
            
            for ticker,s in sns.items():
                #if debug: print(s.latest_quote,s.minute_bar,s.latest_trade)
                curr_results = curr_results.append(pd.DataFrame([[ticker,s.latest_quote.ap,s.latest_quote.bp,s.latest_trade.p,s.minute_bar.c,s.minute_bar.h,s.minute_bar.l,s.minute_bar.o]],columns=['ticker','quote_ap','quote_bp','trade','bar_close','bar_high','bar_low','bar_open']),ignore_index=True)                
            if debug: print(curr_results)
            df_store_data_curr = df_store_data.join(curr_results.set_index('ticker'),on='ticker')
            if debug: print(df_store_data_curr[df_store_data_curr['fit_diff_significance']>5.0].to_string())
            if debug: print(df_store_data_curr[df_store_data_curr['fit_diff_significance']<-5.0].to_string())
            #print(df_store_data_curr.sort_values('fit_diff_significance'))

            # update the significance
            df_store_data_curr['signif_quote_ap'] = (df_store_data_curr['quote_ap'] - df_store_data_curr['fit_expectations']) / df_store_data_curr['stddev']
            if debug: print(df_store_data_curr[df_store_data_curr['signif_quote_ap']>5.0].to_string())
            if debug: print(df_store_data_curr[df_store_data_curr['signif_quote_ap']<-5.0].to_string())
        # check new stocks
        proc_all_tickers = all_tickers
        if len(df)>0 and 'ticker' in df.columns:
            proc_all_tickers = list(df['ticker'].unique())+all_tickers
            proc_all_tickers = set(proc_all_tickers)
        print('Processing: %s' %len(proc_all_tickers))
        iticker=0
        for ticker in proc_all_tickers:
            if debug: print(ticker,iticker)
            iticker+=1
            sys.stdout.flush()

            # if not loaded, then let's compute stuff
            if len(df_store_data[(df_store_data['ticker']==ticker)])==0:

                # checking if it is shortable and tradeable:
                try:
                    aapl_asset = api.get_asset(ticker)
                    #pass
                    #print(aapl_asset) # this is info about it being tradeable
                    #print(aapl_asset.shortable)
                except (alpaca_trade_api.rest.APIError,requests.exceptions.HTTPError,ValueError,urllib3.exceptions.ProtocolError,ConnectionResetError,urllib3.exceptions.ProtocolError,ConnectionResetError,requests.exceptions.ConnectionError) as e:
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
                    except (alpaca_trade_api.rest.APIError,ValueError,urllib3.exceptions.ProtocolError,ConnectionResetError,urllib3.exceptions.ProtocolError,ConnectionResetError,requests.exceptions.ConnectionError) as e:
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
                df_store_data=GetRMSData(daily_prices_60d.index, daily_prices_60d [['adj_close','high','low','open','close']],ticker=ticker,outname='60d',out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_180d.index,daily_prices_180d[['adj_close','high','low','open','close']],ticker=ticker,outname='180d',out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],ticker=ticker,outname='365d',out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_3y.index,  daily_prices_3y  [['adj_close','high','low','open','close']],ticker=ticker,outname='3y',out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_5y.index,  daily_prices_5y  [['adj_close','high','low','open','close']],ticker=ticker,outname='5y', doRelative=False,out_df = df_store_data)
                #df_store_data=GetRMSData(my_index, arr_prices, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[],doJoin=True)
                df_store_data=GetRMSData(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],
                        ticker=ticker,outname='365dspycomparison',spy_comparison = spy_daily_prices_365d[['adj_close','high','low','open','close']],out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_60d.index,daily_prices_60d[['adj_close','high','low','open','close']],
                        ticker=ticker,outname='60dspycomparison',spy_comparison = spy_daily_prices_60d[['adj_close','high','low','open','close']],out_df = df_store_data)
                df_store_data=GetRMSData(daily_prices_5y.index,daily_prices_5y[['adj_close','high','low','open','close']],
                        ticker=ticker,outname='5yspycomparison',spy_comparison = spy_daily_prices_5y[['adj_close','high','low','open','close']],out_df = df_store_data)
        
                # Spy comparison as well as mintue 10day comparison for minute and hour data
                df_store_data=GetRMSData(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                        ticker=ticker,outname='10dhspycomparison', price_key='close',spy_comparison = hour_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True,out_df = df_store_data)
                df_store_data=GetRMSData(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                        ticker=ticker,outname='10dh', price_key='close',out_df = df_store_data)
                df_store_data=GetRMSData(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                        ticker=ticker,outname='10dmspycomparison', price_key='close',spy_comparison = minute_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True,out_df = df_store_data)
                df_store_data=GetRMSData(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                    ticker=ticker,outname='10dm', price_key='close',out_df = df_store_data)
                if debug: print(df_store_data)
            else: # if already loaded, then let's get the historical data
                #sns = api.get_snapshots(['GOOGL','SPY','X'])
                #for s in sns.values():
                #    print(s.latest_quote,s.minute_bar)
                #print(api.get_snapshots(['GOOGL','SPY','X']))
                #print('----')
                #minute_one_price  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=one_day, end=d1)
                #if type(minute_one_price) is not None and len(minute_one_price)>0:
                #    print('%s - most recent price %s' %(ticker,minute_one_price['close'][-1]))
                pass
        # Stalling before the next cycle
        df_store_data.to_csv(outFileName,index=False)
        time.sleep(60)
        
