import os,sys
import datetime
import pandas as pd
import base as b
import math
from ReadData import is_date,SQL_CURSOR,UpdateTable,GetTimeSlot

sqlcursor = SQL_CURSOR(db_name='stocksShort.db')
sqlcursorExtra = SQL_CURSOR(db_name='stocksShortExtra.db')

expectedKeys = ['Index', 'Market Cap', 'Income', 'Sales', 'Book/sh', 'Cash/sh', 'Dividend', 'Dividend %', 'Employees', 'Optionable', 'Shortable', 'Recom', 'P/E', 'Forward P/E', 'PEG', 'P/S', 'P/B', 'P/C', 'P/FCF', 'Quick Ratio', 'Current Ratio', 'Debt/Eq', 'LT Debt/Eq', 'SMA20', 'EPS (ttm)', 'EPS next Y', 'EPS next Q', 'EPS this Y', 'EPS next 5Y', 'EPS past 5Y', 'Sales past 5Y', 'Sales Q/Q', 'EPS Q/Q', 'Earnings', 'SMA50', 'Insider Own', 'Insider Trans', 'Inst Own', 'Inst Trans', 'ROA', 'ROE', 'ROI', 'Gross Margin', 'Oper. Margin', 'Profit Margin', 'Payout', 'SMA200', 'Shs Outstand', 'Shs Float', 'Short Float', 'Short Ratio', 'Target Price', '52W Range', '52W High', '52W Low', 'RSI (14)', 'Rel Volume', 'Avg Volume', 'Volume', 'Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year', 'Perf YTD', 'Beta', 'ATR', 'Volatility', 'Prev Close', 'Price', 'Change']
stock_list=[]
today = datetime.date.today()
todayDateTime=datetime.datetime.now()
all_stocks = b.stock_list+b.etfs
#all_stocks=[['UN',0,0,'NASDAQ',],['X',0,0,'NYSE']]
for iin in all_stocks:
    i=iin[0]
    if i not in stock_list:
        stock_list+=[i]
    else:
        continue
    print(i)
    stock=None
    Load=True
    try:
        stock = pd.read_sql('SELECT * FROM %s' %i, sqlcursorExtra) #,index_col='Date')
        #print(stock.columns)
        stock['LogDate']=pd.to_datetime(stock.LogDate.astype(str), format='%Y-%m-%d')
        stock['LogDate']=pd.to_datetime(stock['LogDate'])
        stock = stock.set_index('LogDate')
        stock = stock.sort_index()
        #print(stock.index[-1])
        if len(stock)>0 and (todayDateTime - stock.index[-1])<datetime.timedelta(days=0,hours=12):
            print('already loaded!')
            Load=False
    except pd.io.sql.DatabaseError:
        pass
    if Load:
        URL = 'https://finviz.com/quote.ashx?t=%s' %i
        filename = '/tmp/%sQUOTE.html' %i
        os.system('wget -O %s %s' %(filename,URL))
        table_MN=None
        try:
            table_MN = pd.read_html(filename)
        except:
            print('error reading file for %s skipping' %i)
            continue
        dataFMap = {}
        for j in table_MN:
        
            if len(j)>0 and len(j[0])>0 and j[0][0].count('Index'):
                for y in range(0,int(len(j.columns)/2)):
                    #print(j[2*y])
                    #print(j[2*y+1])
                    for e in range(0,len(j.values)):
                        #dataFMap+=[[j[2*y][e],j[2*y+1][e].strip().replace('%','').strip() ]]
                        if j[2*y][e] in expectedKeys:
                            dataFMap[j[2*y][e]] = j[2*y+1][e].strip().replace('%','').strip()
        
        if len(dataFMap.values())>0:
            dataFMap['LogDate']=today
            df = pd.DataFrame.from_dict([dataFMap],orient='columns')
            #print(df)
            #print(dataFMap.keys())
            sys.stdout.flush()
            UpdateTable(df,i,sqlcursorExtra) #,index_label='LogDate')
        os.system('rm %s' %filename )


    URL = 'https://www.marketbeat.com/stocks/%s/%s/short-interest/' %(iin[3],i)
    filename = '/tmp/%sv2QUOTE.html' %i
    #print(URL)
    os.system('wget -q -O %s %s' %(filename,URL))
    table_MN=None
    try:
        table_MN = pd.read_html(filename)
        os.system('rm %s' %filename )
    except ValueError:
        newVal='NASDAQ'
        print('wrong value: %s %s' %(iin[3],i))
        if iin[3]=='NASDAQ':
            newVal='NYSE'
        if iin[3]=='NYSE':
            newVal='NASDAQ'
        try:
            URL = 'https://www.marketbeat.com/stocks/%s/%s/short-interest/' %(newVal,i)
            os.system('wget -q -O %s %s' %(filename,URL))
            table_MN = pd.read_html(filename)
            os.system('rm %s' %filename )
        except:
            os.system('rm %s' %filename )
            print('ERROR with %s skipping' %i)
            continue
    m=None
    for t in table_MN:
        #print(t)
        if 'Report Date' in t.columns:
            #print(t)
            #print(t.columns)
            #print(t.head())
            #print(t['Report Date'].values)
            #print(t['Report Date'].str.contains('adsbygoogle'))
    
            # cleaning
            m= t.drop(t[t['Report Date'].str.contains('adsbygoogle')].index)
            m= t.drop(t[t['Report Date'].str.contains('Get the Latest')].index)
            #m= t.drop(t[t['Report Date'].str.strip().contains(' ')].index)
            mi = m['Change from Previous Report'].str.contains('No Change')
            m.loc[mi,'Change from Previous Report'] = 0
            m['Report Date'] = pd.to_datetime(m['Report Date'],errors='coerce')
            m.dropna(subset=['Report Date'],inplace=True)
            for n in t.columns:
                if n.count('Unnamed'):
                    del m[n]
            for v in ['Percentage of Float Shorted','Change from Previous Report']:
                if (v in m.columns) and len(m[v])>0 and (isinstance(m[v].values[0],str) or not math.isnan(m[v].values[0])):
                    #print(m[v])
                    m[v] = m[v].replace({'%':'',',':''}, regex=True).astype('float')
            for v in ['Price on Report Date']:
                m[v] = m[v].replace({'\$':'',',':''}, regex=True).astype('float')
            for v in ['Dollar Volume Sold Short']:
                m[v] = m[v].replace({'\$':'',',':''}, regex=True)
            for v in ['Total Shares Sold Short']:
                m[v] = m[v].replace({' shares':'',',':''}, regex=True).astype('float')
            m = m.sort_values(by=['Report Date'])
            #m = m.rename(columns={'Report Date':'Date'})

    try:
        stock = pd.read_sql('SELECT * FROM %s' %i, sqlcursor) #,index_col='Date')
        print(stock)
        stock['Report Date']=pd.to_datetime(stock['Report Date'].astype(str), format='%Y-%m-%d')
        stock['Report Date']=pd.to_datetime(stock['Report Date'])
        stock = stock.set_index('Report Date')
        stock = stock.sort_index()

        if len(m)>0 and len(stock)>0 and (m['Report Date'].values[-1]-stock.index[-1])<datetime.timedelta(days=0,hours=12):
            print('already loaded new short date %s!' %i)
        else:
            if len(m)>0:
                print(m)
                UpdateTable(m,i,sqlcursor)
    except pd.io.sql.DatabaseError:
        if len(m)>0:
            print(m)
            UpdateTable(m,i,sqlcursor)
    
os.system('rm wget-log*')
os.system('rm *QUOTE.html')
print('DONE')
