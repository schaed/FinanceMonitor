import os,sys,time,datetime
from ReadData import SQL_CURSOR,ALPHA_TIMESERIES,ALPHA_FundamentalData,ConfigTableFromPandas,GetUpcomingEarnings
import pandas  as pd
import numpy as np
import base as b
ReDownload = False
debug=False

# Read and process the overview info
def GetOverview(fd, ticker, connectionCal):
    # should load this once per quarter
    #https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=demo
    overview = fd.get_company_overview(ticker) # has P/E, etc
    if len(overview)>0: # and type(overview) is tuple:
        overview=overview[0]
    # clean
    for dat in ['ExDividendDate','DividendDate','LatestQuarter','LastSplitDate']:
        if dat in overview:
            overview[dat] = pd.to_datetime(overview[dat],errors='coerce')
    for dat in ['Beta', 'RevenueTTM','ForwardPE', 'ForwardAnnualDividendYield', 'RevenuePerShareTTM', 'MarketCapitalization', 'OperatingMarginTTM', 'ShortRatio', 'EPS', 'PayoutRatio', 'PriceToBookRatio', 'CIK', 'GrossProfitTTM', 'PERatio', 'ShortPercentOutstanding', 'ProfitMargin', 'QuarterlyEarningsGrowthYOY', 'TrailingPE', 'SharesShortPriorMonth', 'PEGRatio', '52WeekLow', 'EVToEBITDA',  'PercentInstitutions', 'FullTimeEmployees', 'SharesShort', 'LastSplitFactor', 'ReturnOnAssetsTTM', 'DilutedEPSTTM', 'PriceToSalesRatioTTM', 'SharesFloat', 'EBITDA', '200DayMovingAverage', 'BookValue', 'FiscalYearEnd', 'SharesOutstanding', 'DividendPerShare', 'QuarterlyRevenueGrowthYOY', '52WeekHigh', 'AnalystTargetPrice', 'ShortPercentFloat', '50DayMovingAverage', 'ForwardAnnualDividendRate', 'DividendYield', 'PercentInsiders', 'EVToRevenue', 'ReturnOnEquityTTM']:
        if dat in overview:
            overview[dat] = pd.to_numeric(overview[dat],errors='coerce')

    # Fill the output
    overview['Date']=today=datetime.datetime.now().strftime("%Y-%m-%d")
    overview=overview.set_index('Date')
    if debug: print(overview)

    oEDF = ConfigTableFromPandas('overview',ticker,connectionCal,overview,index_label='Date',tickerName='Symbol')

connectionCal = SQL_CURSOR('earningsCalendar.db')
fd = ALPHA_FundamentalData()

my_3month_calendar=GetUpcomingEarnings(fd,ReDownload)

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
    if j%4==0 and j!=0:
        time.sleep(60)

    # download the overview
    DownloadOverview=False
    if ReDownload: DownloadOverview=True
    if not ReDownload:
        try:
            stockOver = pd.read_sql('SELECT * FROM overview WHERE Symbol="%s"' %(ticker), connectionCal)
            if len(stockOver)==0: DownloadOverview=True
        except:
            DownloadOverview=True
    if DownloadOverview:
        j+=1
        try:
            GetOverview(fd, ticker, connectionCal)
        except:
            print('failed download for %s' %ticker)
            pass

    # try downloading updated info:
    DownloadInfo=False
    if ReDownload:
        DownloadInfo=True
    if not ReDownload:
        try:
            stockInfo = pd.read_sql('SELECT * FROM quarterlyEarnings WHERE ticker="%s"' %(ticker), connectionCal)
            if debug: print(stockInfo)
            if len(stockInfo)==0: DownloadInfo=True
        except:
            DownloadInfo=True
    pastEarnings=[]
    if DownloadInfo:
        j+=1
        try:
            pastEarnings = fd.get_company_earnings(ticker)
        except:
            print('Could not collect: %s' %ticker)
            continue
    else:
        continue


    # Loading the previous earnings!
    try:
        pastEarnings[0]
        if debug: print(pastEarnings[0].keys())
    except:
        print('pastEarnings are empty for %s' %ticker)
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
        
#print(fd.get_balance_sheet_annual(ticker))
#get_company_overview
#get_income_statement_quarterly
# https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=demo
