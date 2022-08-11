import pandas as pd
import numpy as np
from ReadData import ConfigTableFromPandas
# Read and process the overview info
def GetOverview(fd, ticker, connectionCal, debug=False):
    """ GetOverview - Back testing of the SAR trading model. Shows the cumulative return from the strategy
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        connectionCal - sqlite database cursor
    """
    today=datetime.datetime.now().strftime("%Y-%m-%d")
    # should load this once per week?
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
    overview['Date']=today
    overview=overview.set_index('Date')
    if debug: print(overview)

    oEDF = ConfigTableFromPandas('overview',ticker,connectionCal,overview,index_label='Date',tickerName='Symbol')
    return oEDF

def GetStockOverview(fd, ticker, connectionCal, j=0, ReDownload=False, debug=False):
    """ GetStockOverview - Back testing of the SAR trading model. Shows the cumulative return from the strategy
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        connectionCal - sqlite database cursor
        ReDownload - bool - just download info anyway
        debug - bool - print info
        """
    # download the overview
    DownloadOverview=False
    stockOver=[]
    if ReDownload: DownloadOverview=True
    if not ReDownload:
        try:
            stockOver = pd.read_sql('SELECT * FROM overview WHERE Symbol="%s"' %(ticker), connectionCal)
            if len(stockOver)==0: DownloadOverview=True

            # https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest
            # check if the last entry was more than 10 days ago. If so, then load a new entry. Things like the short data are updated once every two weeks
            # info saved as Date when it was recorded
            if 'Date' in stockOver.columns:
                stockOver['Date'] = pd.to_datetime(stockOver['Date'])
                if len(stockOver['Date'])>0:
                    # time check
                    if (np.datetime64(today) - stockOver['Date'].values[-1])>np.timedelta64(10,'D'):
                        DownloadOverview=True
        except:
            DownloadOverview=True
    if DownloadOverview:
        j+=1
        try:
            stockOver = GetOverview(fd, ticker, connectionCal, debug)
        except:
            if debug: print('failed download for %s' %ticker)
            pass

    return stockOver

def GetPastEarnings(fd, ticker, connectionCal, j=0, ReDownload=False, debug=False):
    """ GetPastEarnings - get quarterly and annual earnings
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        connectionCal - sqlite database cursor
        ReDownload - bool - just download info anyway
        debug - bool - print info
        """
    stockInfoQ=[]
    DownloadInfo=False
    if not ReDownload:
        try:
            stockInfoQ = pd.read_sql('SELECT * FROM quarterlyEarnings WHERE ticker="%s"' %(ticker), connectionCal)
            stockInfoA = pd.read_sql('SELECT * FROM annualEarnings WHERE ticker="%s"' %(ticker), connectionCal)
            if debug: print(stockInfoQ)
            if len(stockInfoQ)==0:
                DownloadInfo=True
            else:
                stockInfoQ.sort_values('reportedDate',inplace=True)
                stockInfoA.sort_values('fiscalDateEnding',inplace=True)
                return stockInfoA,stockInfoQ
        except:
            DownloadInfo=True
    else:
        DownloadInfo=True
            
    pastEarnings=[]
    if DownloadInfo:
        #print('download test')
        j+=1
        try:
            pastEarnings = [fd.get_company_earnings(ticker)]
        #print(pastEarnings)
        except:
            print('Could not collect: %s' %ticker)
            return [],[]
    else:
        #continue
        return [],[]

    #
    # Loading the previous earnings!
    #
    annualEarnings=[]
    quarterlyEarnings=[]
    totalDF=[]
    qEDF=[]
    #print('test')
    #print(pastEarnings)
    try:
        pastEarnings[0]
        if debug: print(pastEarnings[0].keys())
    except:
        print('pastEarnings are empty for %s' %ticker)
        #continue
        return [],[]
    if len(pastEarnings)>0 and 'annualEarnings' in pastEarnings[0]:
        annualEarnings = pd.DataFrame(pastEarnings[0]['annualEarnings'])
        try:
            annualEarnings.set_index('fiscalDateEnding')
        except KeyError:
            print('skipping missing fiscaleDate for %s' %ticker)
            return [],[]
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
    if len(pastEarnings)>0 and ('quarterlyEarnings' in pastEarnings[0]):
        quarterlyEarnings = pd.DataFrame(pastEarnings[0]['quarterlyEarnings'])
        # cleaning data
        if ('reportedDate' not in quarterlyEarnings.columns):
            #continue
            print('error...empty quarterly info')
            return [],[]
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
    #print('reached the end')
    return totalDF,qEDF

def GetBalanceSheetQuarterly(fd, ticker, debug=False):
    """ GetBalanceSheetQuarterly - get quarterly balance shee
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        debug bool - print extra info 
    """
    a = fd.get_balance_sheet_quarterly(ticker)
    if len(a)>0:
        a = a[0]
        for sch in a.columns:
            if sch in ['reportedDate','fiscalDateEnding','reportedCurrency']:
                continue
            if sch in a:
                a[sch]=pd.to_numeric(a[sch],errors='coerce')

        a['fiscalDateEnding']=pd.to_datetime(a['fiscalDateEnding'])
        return a
    return []

def GetBalanceSheetAnnual(fd, ticker, debug=False):
    """ GetBalanceSheetAnnual- get annual balance shee
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        debug bool - print extra info 
    """
    a=fd.get_balance_sheet_annual(ticker)
    if len(a)>0:
        a = a[0]
        a['fiscalDateEnding']=pd.to_datetime(a['fiscalDateEnding'])
        for sch in a.columns:
            if sch in ['reportedDate','fiscalDateEnding','reportedCurrency']:
                continue
            if sch in a:
                a[sch]=pd.to_numeric(a[sch],errors='coerce')
        
        return a
    return []

def GetIncomeStatement(fd, ticker, annual=False, debug=False):
    """ GetIncomeStatue- get the income statement
        
         Parameters:
         fd - Fundamental data api source from alpha vantage
         ticker - str
                Stock ticker symbol
        annual - bool - annual or quarterly when false
        debug - bool - print extra info 
    """

    if annual:
        a = fd.get_income_statement_annual(ticker)
        if len(a)>0:
            a = a[0]
            for sch in a.columns:
                if sch in ['reportedDate','fiscalDateEnding','reportedCurrency']:
                    continue
                if sch in a:
                    a[sch]=pd.to_numeric(a[sch],errors='coerce')

            a['fiscalDateEnding']=pd.to_datetime(a['fiscalDateEnding'])

            return a
    else:
        a = fd.get_income_statement_quarterly(ticker)
        if len(a)>0:
            a = a[0]
            for sch in a.columns:
                if sch in ['reportedDate','fiscalDateEnding','reportedCurrency']:
                    continue
                if sch in a:
                    a[sch]=pd.to_numeric(a[sch],errors='coerce')

            a['fiscalDateEnding']=pd.to_datetime(a['fiscalDateEnding'])

            return a        
    return []

