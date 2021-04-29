from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
import alpaca_trade_api
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from techindicators import techindicators
import talib
import datetime,time
import pandas as pd
import numpy as np
import os
import sqlite3
from dateutil.parser import parse

# db_name the database name
def SQL_CURSOR(db_name='stocksAV.db'):

    connection = sqlite3.connect(db_name)
    #cursor = connection.cursor()
    return connection

# append this entry to the sql table
def UpdateTable(stock, ticker, sqlcursor, index_label='Date'):
    if index_label!=None:
        stock.to_sql(ticker, sqlcursor,if_exists='append', index=True, index_label=index_label)
    else:
        stock.to_sql(ticker, sqlcursor,if_exists='append', index=True)        

# try to read back info. if not, then update the SQL database
def ConfigTableFromPandas(tableName, ticker, sqlcursor, earnings, index_label='Date',tickerName='ticker'):

    stock = None
    try:
        stock = pd.read_sql('SELECT * FROM %s WHERE %s="%s"' %(tableName,tickerName,ticker), sqlcursor) #,index_col='Date')
        if len(stock)==0 and len(earnings)>0: # if empty, then let's fill it
            UpdateTable(earnings, tableName, sqlcursor, index_label=None)
            return earnings
        # if not empty, then we need to merge
        stock[index_label]=pd.to_datetime(stock[index_label].astype(str), format='%Y-%m-%d')
        stock[index_label]=pd.to_datetime(stock[index_label])
        stock = stock.set_index(index_label)
        stock = stock.sort_index()
        today=datetime.datetime.now()
        if len(earnings)>0:
            earnings[index_label]=pd.to_datetime(earnings[index_label])
            today = earnings.sort_values(index_label)[index_label].values[-1]
            print(today)
        StartLoading = True
        if (today - stock.index[-1])>datetime.timedelta(days=1,hours=12) and (StartLoading):
            stockCompact=earnings
            # make sure we only add newer dates
            stockCompact = GetTimeSlot(stockCompact, days=(today - stock.index[-1]).days)
            UpdateTable(stockCompact, tableName, sqlcursor, index_label=index_label)
            stock = pd.concat([stock,stockCompact])
            stock.sort_index()
    except:
        print('%s is a new entry to the database....' %tableName)
        stock=earnings
        UpdateTable(stock, tableName, sqlcursor, index_label=None)

    return stock

# try to read back info. if not, then update the SQL database
def ConfigTable(ticker, sqlcursor, ts, readType, j=0, index_label='Date',hoursdelay=5):

    stock = None
    try:
        stock = pd.read_sql('SELECT * FROM %s' %ticker, sqlcursor) #,index_col='Date')
        stock[index_label]=pd.to_datetime(stock[index_label].astype(str), format='%Y-%m-%d')
        stock[index_label]=pd.to_datetime(stock[index_label])
        stock = stock.set_index(index_label)
        stock = stock.sort_index()
        today=datetime.datetime.now()
        StartLoading = True
        if stock.index[-1].weekday()==4 and (today - stock.index[-1])<datetime.timedelta(days=4):
            StartLoading=False
        if (today - stock.index[-1])>datetime.timedelta(days=1,hours=hoursdelay) and (StartLoading):
            try:
                stockCompact=runTickerAlpha(ts,ticker,'compact')
                j+=1
            except ValueError:
                print('%s could not load compact....' %ticker)
                j+=1
                return [],j
            # make sure we only add newer dates
            stockCompact = GetTimeSlot(stockCompact, days=(today - stock.index[-1]).days)
            UpdateTable(stockCompact, ticker, sqlcursor, index_label=index_label)
            stock = pd.concat([stock,stockCompact])
            stock = stock.sort_index()
    except:
        print('%s is a new entry to the database....' %ticker)
        try:
            stock=runTickerAlpha(ts,ticker,'full')
            j+=1
        except ValueError:
            j+=1
            print('%s could not load....' %ticker)
            return [],j
        UpdateTable(stock, ticker, sqlcursor, index_label=index_label)

    return stock,j

def ALPACA_REST():
    ALPACA_ID = os.getenv('ALPACA_ID')
    ALPACA_PAPER_KEY = os.getenv('ALPACA_PAPER_KEY')
    api = REST(ALPACA_ID,ALPACA_PAPER_KEY)
    return api

def IS_ALPHA_PREMIUM():
    
    ALPHA_PREMIUM = os.getenv('ALPHA_PREMIUM')
    if ALPHA_PREMIUM=='' or ALPHA_PREMIUM==None: return False
    if int(ALPHA_PREMIUM)==1:
        ALPHA_PREMIUM=True
    else:
        ALPHA_PREMIUM=False
    return ALPHA_PREMIUM

def IS_ALPHA_PREMIUM_WAIT_ITER():
    if IS_ALPHA_PREMIUM():
        return 30 # can update to 75 or 74
    return 4

def ALPHA_TIMESERIES():
    ALPHA_ID = os.getenv('ALPHA_ID')
    ts = TimeSeries(key=ALPHA_ID)
    return ts
 
def ALPHA_FundamentalData(output_format='pandas'):#pandas, json, csv, csvpan
    ALPHA_ID = os.getenv('ALPHA_ID')
    fd = FundamentalData(key=ALPHA_ID,output_format=output_format)
    return fd

def GetTimeSlot(stock, days=365, startDate=None):
    today=datetime.datetime.now()
    if startDate!=None:
        today = startDate
    past_date = today + datetime.timedelta(days=-1*days)
        
    date=stock.truncate(before=past_date, after=startDate)
    #date = stock[nearest(stock.index,past_date)]
    return date

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

# detail reads the amount of data compact or full
def runTickerAlpha(ts, ticker, detail='full'): 
    
    a=ts.get_daily_adjusted(ticker,detail)
    a_new={}
    cols = ['Date','open','high','low','close','volume']
    cols = ['Date','open','high','low','close','adj_close','volume','dividendamt','splitcoef']
    my_floats = ['open','high','low','close']
    my_floats = ['open','high','low','close','adj_close','volume','dividendamt','splitcoef']
    
    #'5. adjusted close', '6. volume', '7. dividend amount', '8. split coefficient'
    for ki in cols:
        a_new[ki]=[]
    
    for entry in a:
        for key,i in entry.items():
            if not is_date(key):
                continue
            #print(key)
            a_new['Date']+=[key]
            ij=0
            todays_values = list(i.values())
            for j in ['open','high','low','close','adj_close','volume','dividendamt','splitcoef']:
                a_new[j]+=[todays_values[ij]]
                ij+=1
    # format
    output = pd.DataFrame(a_new)
    output['Date']=pd.to_datetime(output['Date'].astype(str), format='%Y-%m-%d')
    output['Date']=pd.to_datetime(output['Date'])
    output[my_floats]=output[my_floats].astype(float)
    output['volume'] = output['volume'].astype(np.int64)
    output = output.set_index('Date')
    output = output.sort_index()
    return output

# Alpaca info
def runTicker(api, ticker, timeframe=TimeFrame.Day, start=None, end=None):
    today=datetime.datetime.now()
    if timeframe==TimeFrame.Day and start==None and end==None:
        yesterday = today + datetime.timedelta(days=-1)
        d1 = yesterday.strftime("%Y-%m-%d")
        fouryao = (today + datetime.timedelta(days=-364*4.5)).strftime("%Y-%m-%d")
        trade_days = api.get_bars(ticker, timeframe, fouryao, d1, 'raw').df
    elif start!=None and end!=None:
        #start_date = ''
        trade_days = api.get_bars(ticker, timeframe, start=start, end=end, adjustment='raw').df
    return trade_days

# get various types
#   api.get_bars
#   api.get_quotes
#   api.get_trades
def runTickerTypes(api, ticker, timeframe=TimeFrame.Day, start=None, end=None, limit=None, dataType='bars'):
    today=datetime.datetime.now()
    trade_days=None
    if timeframe==TimeFrame.Day and start==None and end==None:
        yesterday = today + datetime.timedelta(days=-1)
        d1 = yesterday.strftime("%Y-%m-%d")
        fouryao = (today + datetime.timedelta(days=-364*4.5)).strftime("%Y-%m-%d")
        trade_days = api(ticker, timeframe=timeframe,
                        start=fouryao, end=d1,  limit=limit).df #adjustment='raw',
            
    elif start!=None and end!=None:
        start_date = ''
        if dataType=='bars':
            trade_days = api(ticker, timeframe=timeframe,
                            start=start, end=end, adjustment='raw', limit=limit).df 
        else:
            #print(ticker)
            trade_days = api(ticker, start=start, end=end, limit=limit).df             
    return trade_days

# add stock data and market data to compute metrics
def AddInfo(stock,market,debug=False):
    # let's make sure we sort this correctly
    #stock = stock.sort_index()    
    #print(stock.tail())
    # Label Volume as positive or negative
    stock['pos_volume'] = 0
    #stock.loc[stock.open>=stock.close,'pos_volume'] = stock.volume
    #stock.loc[stock.open<stock.close,'neg_volume'] = stock.volume
    stock.loc[stock.adj_close>=stock.adj_close.shift(1),'pos_volume'] = stock.volume
    stock.loc[stock.adj_close<stock.adj_close.shift(1),'neg_volume'] = stock.volume
    # SMA
    stock['sma10']=techindicators.sma(stock['adj_close'],10)
    stock['sma20']=techindicators.sma(stock['adj_close'],20)
    if len(stock['adj_close'])>50:
        stock['sma50']=techindicators.sma(stock['adj_close'],50)
    else: stock['sma50']=np.zeros(len(stock['adj_close']))
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
    
    # add various future returns
    stock['fiveday_future_return']=stock['adj_close'].shift(-5).pct_change(periods=5)
    stock['oneday_future_return']=stock['adj_close'].shift(-1).pct_change(periods=1)
    stock['twoday_future_return']=stock['adj_close'].shift(-2).pct_change(periods=2)
    stock['thrday_future_return']=stock['adj_close'].shift(-5).pct_change(periods=3)
    stock['fiveday_prior_vix']=stock['adj_close'].rolling(5).std()
    stock['oneday_prior_vix']=stock['adj_close'].rolling(1).std()
    stock['twoday_prior_vix']=stock['adj_close'].rolling(2).std()
    stock['thrday_prior_vix']=stock['adj_close'].rolling(3).std()
    #print(stock[['adj_close','oneday_future_return','twoday_future_return']])
    
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
    stock['bop']=techindicators.bop(stock['high'],stock['low'],stock['close'],stock['open'],14)
    #stock['pdmd'],stock['ndmd'],stock['adx']=techindicators.adx(stock['high'],stock['low'],stock['close'],14)
    stock['HT_DCPERIOD']=talib.HT_DCPERIOD(stock['adj_close']) 
    stock['HT_DCPHASE']=talib.HT_DCPHASE(stock['adj_close']) 
    stock['HT_TRENDMODE']=talib.HT_TRENDMODE(stock['adj_close']) 
    stock['HT_SINE'],stock['HT_SINElead']=talib.HT_SINE(stock['adj_close'])
    stock['HT_PHASORphase'],stock['HT_PHASORquad']=talib.HT_PHASOR(stock['adj_close'])     
    stock['adx']=talib.ADX(stock['high'],stock['low'],stock['close'],14) 
    stock['willr']=talib.WILLR(stock['high'],stock['low'],stock['close'],14) 
    stock['ultosc']=talib.ULTOSC(stock['high'],stock['low'],stock['close'],timeperiod1=7, timeperiod2=14, timeperiod3=28) 
    stock['adx']=talib.ADX(stock['high'],stock['low'],stock['close'],14) 
    stock['aroonUp'],stock['aroonDown'],stock['aroon']=techindicators.aroon(stock['high'],stock['low'],25)
    stock['senkou_spna_A'],stock['senkou_spna_B'],stock['chikou_span']=techindicators.IchimokuCloud(stock['high'],stock['low'],stock['adj_close'],9,26,52)
    stock['SAR'] = talib.SAR(stock.high, stock.low, acceleration=0.02, maximum=0.2)
    stock['vwap14']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],14)
    stock['vwap10']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],10)
    stock['vwap20']=techindicators.vwap(stock['high'],stock['low'],stock['close'],stock['volume'],20)
    stock['chosc']=techindicators.chosc(stock['high'],stock['low'],stock['close'],stock['volume'],3,10)
    stock['market'] = market['adj_close']
    stock['corr14']=stock['adj_close'].rolling(14).corr(market['market'])
    
    stock['weekly_return']=stock['adj_close'].pct_change(freq='W')
    stock['monthly_return']=stock['adj_close'].pct_change(freq='M')
    stock_1y = GetTimeSlot(stock)
    if len(stock_1y['adj_close'])<1:
        print('Ticker has no adjusted close info.')
        stock['yearly_return']=stock['adj_close']
    else:
        stock['yearly_return']=stock['adj_close']/stock_1y['adj_close'][0]-1
    stock['CDLABANDONEDBABY']=talib.CDLABANDONEDBABY(stock['open'],stock['high'],stock['low'],stock['close'], penetration=0)
    if len(stock)>2:
        for ky in talib.__dict__.keys():
            if 'CDL' in ky and not 'stream' in ky:
                stock[ky]=talib.__dict__[ky](stock['open'],stock['high'],stock['low'],stock['close'])
    end = time.time() 
    
    if debug: print('Process time to new: %s' %(end - start))

# collect the upcoming earnings info
# fd in the fundamentals data api
# ReDownload: newly downloads the file when True.
def GetUpcomingEarnings(fd,ReDownload):

    if os.path.exists('stockEarnings.csv') and not ReDownload:
        my_3month_calendar = pd.read_csv('stockEarnings.csv')
    else:
        my_3month_calendar = fd.get_earnings_calendar('3month')
        if len(my_3month_calendar)>0:
            my_3month_calendar = my_3month_calendar[0]
            my_3month_calendar.to_csv('stockEarnings.csv')
    
    # clean up
    my_3month_calendar['reportDate']=pd.to_datetime(my_3month_calendar['reportDate'])
    my_3month_calendar['fiscalDateEnding']=pd.to_datetime(my_3month_calendar['fiscalDateEnding'])
    my_3month_calendar['estimate']=pd.to_numeric(my_3month_calendar['estimate'])
    my_3month_calendar=my_3month_calendar.set_index('reportDate')
    my_3month_calendar=my_3month_calendar.sort_index()
    return my_3month_calendar
