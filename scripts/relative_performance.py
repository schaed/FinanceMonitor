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

def GetRMSData(my_index, arr_prices, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[],doRelative=False,doJoin=True, out_df = [],alt_ticker=''):
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
    
    start = time.time()
    prices = arr_prices[price_key]
    x = mdates.date2num(my_index)
    #xx = np.linspace(x.min(), x.max(), 1000)
    #dd = mdates.num2date(xx)
    
    # prepare a spy comparison
    if len(spy_comparison)>0:
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

            #arr_prices = arr_prices.join(spy_comparison,how='left',rsuffix='_spy')
            it_list = ['high','low','open','close']
            if price_key not in it_list: it_list += [price_key]
            for i in it_list:
                if len(arr_prices[i+'_spy'])>0:
                    arr_prices[i] /= (arr_prices[i+'_spy'] / arr_prices[i+'_spy'].iloc[-1])
            prices = arr_prices[price_key]

    end = time.time()
    if debug: print('Process time to JOIN: %s' %(end - start))
    start = time.time()    
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

    out_arr = [p4(x)[-1],stddev,diff[-1],prices[-1],alt_ticker]
    output_lines = '%s,%s,%s,%s' %(p4(x)[-1],stddev,diff[-1],prices[-1])
    end = time.time()
    if debug: print('Process time to get Fit data: %s' %(end - start))
    if stddev!=0.0:
        output_lines = '%0.3f,%0.3f,%0.3f,%s,%s' %(p4(x)[-1],stddev,diff[-1]/stddev,prices[-1],alt_ticker)
        out_arr = [p4(x)[-1],stddev,diff[-1]/stddev,prices[-1],alt_ticker]    
    if debug: print('%s,%s,%s' %(ticker,outname,output_lines))
    # check if stuff is filled
    if len(out_df[(out_df['ticker']==ticker) & (out_df['time_span']==outname)])==0:
        out_df = out_df.append(pd.DataFrame([[ticker,outname]+out_arr],columns=['ticker','time_span','fit_expectations','stddev','fit_diff_significance','current_price','alt_ticker']),ignore_index=True)
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
    outFileName='News/correl_%s_%s_%s.csv' %(today.day,today.month,today.year)
    df=[]
    todayFilter = (today + datetime.timedelta(days=-1*filter_shift_days))
    d1 = todayFilter.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    thirty_days = (todayFilter + datetime.timedelta(days=-30)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    one_day = (todayFilter + datetime.timedelta(days=-1)).strftime("%Y-%m-%dT%H:%M:%S-05:00")

    # create a dataframe to store all of the info
    curr_results = pd.DataFrame(columns=['ticker','quote','trade','bar_close','bar_high','bar_low','bar_open'])
    df_store_data = pd.DataFrame(columns=['ticker','time_span','fit_expectations','stddev','fit_diff_significance','current_price','alt_ticker'])
    if os.path.exists(outFileName):
        df_store_data = pd.read_csv(outFileName)
        
    # load ETFs
    all_tickers = []
    for etf in base.etfs:
        all_tickers +=[etf[0]]
    # check new stocks
    proc_all_tickers = all_tickers
    if debug: print('Processing: %s' %len(proc_all_tickers))
    iticker=0
    for ticker in proc_all_tickers:
        iticker+=1

        # if not loaded, then let's compute stuff
        if len(df_store_data[(df_store_data['ticker']==ticker)])==0:
            daily_prices,j    = ConfigTable(ticker, sqlcursor,ts,'full',hoursdelay=18)
            if filter_shift_days>0:
                daily_prices  = GetTimeSlot(daily_prices, days=6*365, startDate=todayFilter)
            daily_prices_60d  = GetTimeSlot(daily_prices, days=60+filter_shift_days)
            daily_prices_180d = GetTimeSlot(daily_prices, days=180+filter_shift_days)
            daily_prices_365d = GetTimeSlot(daily_prices, days=365+filter_shift_days)
            daily_prices_3y   = GetTimeSlot(daily_prices, days=3*365+filter_shift_days)
            daily_prices_5y   = GetTimeSlot(daily_prices, days=5*365+filter_shift_days)
            daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)
            #if debug: print('ticker,time_span,fit_expectations,stddev,fit_diff_significance,current_price')
        
            # Run:
            df_store_data=GetRMSData(daily_prices_60d.index, daily_prices_60d [['adj_close','high','low','open','close']],ticker=ticker,outname='60d',out_df = df_store_data)
            df_store_data=GetRMSData(daily_prices_180d.index,daily_prices_180d[['adj_close','high','low','open','close']],ticker=ticker,outname='180d',out_df = df_store_data)
            df_store_data=GetRMSData(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],ticker=ticker,outname='365d',out_df = df_store_data)
            df_store_data=GetRMSData(daily_prices_3y.index,  daily_prices_3y  [['adj_close','high','low','open','close']],ticker=ticker,outname='3y',out_df = df_store_data)
            df_store_data=GetRMSData(daily_prices_5y.index,  daily_prices_5y  [['adj_close','high','low','open','close']],ticker=ticker,outname='5y', doRelative=False,out_df = df_store_data)

            # iterate through alternatives
            for alt_ticker in proc_all_tickers:
                start = time.time()
                spy,j    = ConfigTable(alt_ticker, sqlcursor,ts,'full',hoursdelay=18)

                daily_prices_after = daily_prices.copy(True)
                #spy = spy.copy(True)
                daily_prices_after = daily_prices.join(spy,how='left',rsuffix='_spy')
                daily_prices_after_60d  = GetTimeSlot(daily_prices_after, days=60+filter_shift_days)
                daily_prices_after_180d = GetTimeSlot(daily_prices_after, days=180+filter_shift_days)
                daily_prices_after_365d = GetTimeSlot(daily_prices_after, days=365+filter_shift_days)
                daily_prices_after_3y   = GetTimeSlot(daily_prices_after, days=3*365+filter_shift_days)
                daily_prices_after_5y   = GetTimeSlot(daily_prices_after, days=5*365+filter_shift_days)
                
                end = time.time()
                if debug: print('Process time to read SQL or ask API: %s' %(end - start))
                start = time.time()
                if filter_shift_days>0:
                    spy  = GetTimeSlot(spy, days=6*365, startDate=todayFilter)
                spy_daily_prices_60d  = GetTimeSlot(spy, days=60+filter_shift_days)
                spy_daily_prices_365d = GetTimeSlot(spy, days=365+filter_shift_days)
                spy_daily_prices_5y   = GetTimeSlot(spy, days=5*365+filter_shift_days)
                #if debug:
                print(ticker,alt_ticker,iticker)
                sys.stdout.flush()
                end = time.time()
                if debug: print('Process time to getTimeSlots: %s' %(end - start))
                # if not loaded, then let's compute stuff
                start = time.time()
                if len(spy)>0:
                    
                    # Correlate
                    df_store_data=GetRMSData(daily_prices_after_60d.index,daily_prices_after_60d,
                            ticker=ticker,outname='60d'+alt_ticker+'comparison',spy_comparison = spy_daily_prices_60d[['adj_close','high','low','open','close']],out_df = df_store_data, alt_ticker=alt_ticker)
                    df_store_data=GetRMSData(daily_prices_after_365d.index,daily_prices_after_365d,
                            ticker=ticker,outname='365d'+alt_ticker+'comparison',spy_comparison = spy_daily_prices_365d[['adj_close','high','low','open','close']],out_df = df_store_data, alt_ticker=alt_ticker)
                    df_store_data=GetRMSData(daily_prices_after_5y.index,daily_prices_after_5y,
                        ticker=ticker,outname='5ys'+alt_ticker+'comparison',spy_comparison = spy_daily_prices_5y[['adj_close','high','low','open','close']],out_df = df_store_data, alt_ticker=alt_ticker)
                    end = time.time()
                    if debug: print('Process time to get RMS data: %s' %(end - start))
        
        df_store_data.to_csv(outFileName,index=False)
        #time.sleep(60)
        
