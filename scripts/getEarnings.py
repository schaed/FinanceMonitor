import os,sys,time
from ReadData import SQL_CURSOR,ALPHA_TIMESERIES,ALPHA_FundamentalData,ConfigTableFromPandas
import pandas  as pd
import numpy as np
import base as b
ReDownload = False
debug=False

connectionCal = SQL_CURSOR('earningsCalendar.db')
fd = ALPHA_FundamentalData()
if os.path.exists('stockEarnings.csv') and not ReDownload:
    my_3month_calendar = pd.read_csv('stockEarnings.csv')
else:
    my_3month_calendar = fd.get_earnings_calendar('3month')
    if len(my_3month_calendar)>0:
        my_3month_calendar = my_3month_calendar[0]
        my_3month_calendar.to_csv('stockEarnings.csv')
#print(fd.get_company_earnings_calendar('IBM','3month'))
#my_3month_calendar
if debug: print(my_3month_calendar.columns)
# clean up
my_3month_calendar['reportDate']=pd.to_datetime(my_3month_calendar['reportDate'])
my_3month_calendar['fiscalDateEnding']=pd.to_datetime(my_3month_calendar['fiscalDateEnding'])
my_3month_calendar['estimate']=pd.to_numeric(my_3month_calendar['estimate'])
my_3month_calendar=my_3month_calendar.set_index('reportDate')
my_3month_calendar=my_3month_calendar.sort_index()
if debug: print(my_3month_calendar)
if debug: print(my_3month_calendar['symbol'])
#print(fd.get_company_overview('IBM'))
#print(fd.get_company_earnings('IBM'))
#print(fd.get_earnings_calendar('3month'))
#print(fd.get_company_earnings_calendar('IBM','3month'))
#print(fd.get_income_statement_quarterly('IBM'))
#cursorCal = connectionCal.cursor()

j=0
tickers=['IBM']
#tickers = my_3month_calendar['symbol'].values
for t in b.stock_list:
    if t[0].count('^'):
            continue
    if t[0] not in tickers:
        tickers+=[t[0]]
print('Processing %s tickers' %(len(tickers)))
for ticker in tickers:
    print(ticker)
    sys.stdout.flush()
    if j%4==0 and j!=0:
        time.sleep(59)
    try:
        pastEarnings = fd.get_company_earnings(ticker)
    except:
        j+=1
        print('Could not collect: %s' %ticker)
        continue
    if debug:
        if len(pastEarnings)>0: print(pastEarnings[0].keys())
    else:
        j+=1
        continue
    if len(pastEarnings)>0 and 'annualEarnings' in pastEarnings[0]:
        annualEarnings = pd.DataFrame(pastEarnings[0]['annualEarnings'])
        annualEarnings.set_index('fiscalDateEnding')
        if debug: print(annualEarnings.dtypes)
        # cleaning up data
        annualEarnings['ticker'] = np.array([ticker for _ in range(0,len(annualEarnings))])
        annualEarnings['fiscalDateEnding']=pd.to_datetime(annualEarnings['fiscalDateEnding'])
        for sch in ['reportedEPS']:
            if sch in annualEarnings:
                annualEarnings[sch]=pd.to_numeric(annualEarnings[sch],errors='coerce')
        totalDF = ConfigTableFromPandas('annualEarnings',ticker,connectionCal,annualEarnings,index_label='fiscalDateEnding')
        if debug:
            print(annualEarnings)
            print(totalDF)
    if len(pastEarnings)>0 and 'quarterlyEarnings' in pastEarnings[0]:
        quarterlyEarnings = pd.DataFrame(pastEarnings[0]['quarterlyEarnings'])
        # cleaning data
        quarterlyEarnings['ticker'] = np.array([ticker for _ in range(0,len(quarterlyEarnings))])
        quarterlyEarnings.set_index('reportedDate')
        quarterlyEarnings['reportedDate']=pd.to_datetime(quarterlyEarnings['reportedDate'])
        quarterlyEarnings['fiscalDateEnding']=pd.to_datetime(quarterlyEarnings['fiscalDateEnding'])
        for sch in ['surprise','reportedEPS','estimatedEPS','surprisePercentage']:
            if sch in quarterlyEarnings:
                quarterlyEarnings[sch]=pd.to_numeric(quarterlyEarnings[sch],errors='coerce')
        if debug:
            print(quarterlyEarnings)
            print(quarterlyEarnings.dtypes)
        qEDF = ConfigTableFromPandas('quarterlyEarnings',ticker,connectionCal,quarterlyEarnings,index_label='reportedDate')
        if debug: print(qEDF)
    j+=1
        
#print(fd.get_balance_sheet_annual(ticker))
#get_company_overview
#get_income_statement_quarterly
# https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=demo
