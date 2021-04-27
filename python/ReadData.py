from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import REST
import alpaca_trade_api
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import datetime
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
def ConfigTableFromPandas(tableName, ticker, sqlcursor, earnings, index_label='Date'):

    stock = None
    try:
        stock = pd.read_sql('SELECT * FROM %s WHERE ticker="%s"' %(tableName,ticker), sqlcursor) #,index_col='Date')
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
def ConfigTable(ticker, sqlcursor, ts, readType, j=0, index_label='Date'):

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
        if (today - stock.index[-1])>datetime.timedelta(days=1,hours=12) and (StartLoading):
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
            stock.sort_index()
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

def ALPHA_TIMESERIES():
    ALPHA_ID = os.getenv('ALPHA_ID')
    ts = TimeSeries(key=ALPHA_ID)
    return ts
 
def ALPHA_FundamentalData(output_format='pandas'):#pandas, json, csv, csvpan
    ALPHA_ID = os.getenv('ALPHA_ID')
    fd = FundamentalData(key=ALPHA_ID,output_format=output_format)
    return fd

def GetTimeSlot(stock, days=365):
    today=datetime.datetime.now()
    past_date = today + datetime.timedelta(days=-1*days)
    date=stock.truncate(before=past_date)
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
