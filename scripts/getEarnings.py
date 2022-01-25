import os,sys,time,datetime
from ReadData import SQL_CURSOR,ALPHA_TIMESERIES,ALPHA_FundamentalData,ConfigTableFromPandas,GetUpcomingEarnings
from Earnings import GetStockOverview,GetPastEarnings
import pandas  as pd
import numpy as np
import base as b
ReDownload = False
debug=False


connectionCal = SQL_CURSOR('earningsCalendarv2.db')
# clean up the null results before we start
connectionCal.cursor().execute('DELETE FROM quarterlyEarnings WHERE (reportedDate>="2021-04-20" AND reportedEPS is NULL)')
# remove the duplicates
#connectionCal.cursor().execute("DELETE FROM quarterlyEarnings WHERE rowid NOT IN (  SELECT MIN(rowid)   FROM quarterlyEarnings   GROUP BY reportedDate,ticker )").fetchall()
fd = ALPHA_FundamentalData()

my_3month_calendar=GetUpcomingEarnings(fd,True)

if debug: print(my_3month_calendar.columns)
if debug: print(my_3month_calendar)
if debug: print(my_3month_calendar['symbol'])

#print(fd.get_company_earnings('IBM'))
#print(fd.get_earnings_calendar('3month'))
#print(fd.get_company_earnings_calendar('IBM','3month'))
# should load this
#https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=demo
#print(fd.get_income_statement_quarterly('IBM'))
#ov = fd.get_company_overview('IBM')
#if len(ov)>0 and type(ov) is tuple:
#        ov=ov[0]
#print(ov)
#cursorCal = connectionCal.cursor()
#sys.exit(0)
j=0
tickers=['IBM']
today=datetime.datetime.now().strftime("%Y-%m-%d")
tickers = my_3month_calendar['symbol'].values.tolist()
for t in b.stock_list:
    if t[0].count('^'):
            continue
    if t[0] not in tickers:
        tickers+=[t[0]]
print('Processing %s tickers' %(len(tickers)))
#sys.exit(0)
for ticker in tickers:
    
    print(ticker)
    sys.stdout.flush()
    #if j%4==0 and j!=0:
    if j%70==0 and j!=0:
        time.sleep(60)

    # Load the stock overview
    stockOver = GetStockOverview(fd, ticker, connectionCal, ReDownload=ReDownload, debug=debug):
    
    # try downloading updated info:
    DownloadInfo=False
    if ReDownload:
        DownloadInfo=True

    reportDate = my_3month_calendar[my_3month_calendar['symbol']==ticker].index.values
    if len(reportDate)>0:
        if (np.datetime64(today) - reportDate[0])>np.timedelta64(5,'h'):
            print('will update earnings for %s' %ticker)
            DownloadInfo=True

    # Collect past earnings
    GetPastEarnings(fd, ticker, connectionCal, ReDownload=ReDownload, debug=debug)

#print(fd.get_balance_sheet_annual(ticker))
#get_company_overview
#get_income_statement_quarterly
# https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=demo
