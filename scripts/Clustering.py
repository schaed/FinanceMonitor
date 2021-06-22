from dataclasses import dataclass
from ReadData import SQL_CURSOR,ALPHA_FundamentalData,ConfigTable,ALPHA_TIMESERIES,AddInfo,GetTimeSlot,ALPACA_REST,runTicker,getQuotesTS,getQuotes
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm1
import pytz,os,sys
import base as b
from sklearn.cluster import KMeans
debug=False
draw=True

# Map of industry to an integer
def BuildIndustryMap(industry,ind_map):
    ind_key=0
    if industry in ind_map:
        return ind_map[industry]
    else:
        if len(ind_map.keys())>0:
            ind_key = max(ind_map.values())+1
        ind_map[industry] = ind_key
    return ind_key;

def CollectEarnings(ticker,conn):
    """CollectEarnings
         Parameters:
             ticker:            str   = None # ticker symbol
             conn:              sqlite connection
    """
    stockInfoQuarter=None
    stockInfoAnnual=None
    company_overview = []
    try:
        company_overview = pd.read_sql('SELECT * FROM overview WHERE Symbol="%s"' %(ticker), conn)
        stockInfoQuarter = pd.read_sql('SELECT * FROM quarterlyEarnings WHERE ticker="%s"' %(ticker), conn)
        stockInfoAnnual = pd.read_sql('SELECT * FROM annualEarnings WHERE ticker="%s"' %(ticker), conn)
        stockInfoAnnual['fiscalDateEnding']=pd.to_datetime(stockInfoAnnual['fiscalDateEnding'])        
        stockInfoAnnual.set_index('fiscalDateEnding',inplace=True)
        stockInfoQuarter['reportedDate']=pd.to_datetime(stockInfoQuarter['reportedDate'],errors='coerce')
        stockInfoQuarter.set_index('reportedDate',inplace=True)
        stockInfoQuarter['fiscalDateEnding']=pd.to_datetime(stockInfoQuarter['fiscalDateEnding'],errors='coerce')
        
        if len(company_overview)>0:
            for d in company_overview.columns:
                if d not in ['Symbol','AssetType','Name','Description','CIK','Exchange','Currency','Country','Sector','Industry','Address','FiscalYearEnd','LatestQuarter','DividendDate','ExDividendDate','LastSplitFactor','LastSplitDate']:
                    company_overview[d]=pd.to_numeric(company_overview[d],errors='coerce')
            for d in ['LatestQuarter','DividendDate','ExDividendDate','LastSplitFactor','LastSplitDate']:
                company_overview[d]=pd.to_datetime(company_overview[d],errors='coerce')
            company_overview.index = pd.to_datetime(company_overview.index,errors='coerce')
    except (pd.io.sql.DatabaseError,KeyError):
        print('ERROR collecting earnings history for %s' %ticker)
        pass
    if debug:
        print(stockInfoQuarter)
        print(stockInfoQuarter.dtypes)
        print(stockInfoAnnual)
        print(stockInfoAnnual.dtypes)
        print(company_overview)
        print(company_overview.dtypes)
    return stockInfoQuarter,stockInfoAnnual,company_overview

if __name__ == "__main__":
    # execute only if run as a script
    connectionCal = SQL_CURSOR('earningsCalendar.db')
    fd = ALPHA_FundamentalData()
    sqlcursor = SQL_CURSOR()
    ts = ALPHA_TIMESERIES()
    api = ALPACA_REST()
    ticker='MPC'
    ind_map={}
    sec_map={}
    doEarnings = True
    #stockInfoQuarter,stockInfoAnnual,company_overview=CollectEarnings(ticker,connectionCal)
    readType='full'
    j=0
    data_points = ['day_return','day2_return','day3_return','day4_return','day5_return','day15_return','30d_return','60d_return','180d_return','volatitilty','5d_vol','30d_vol','180d_vol']
    earn_points = ['ShortPercentFloat','PercentInsiders','PercentInstitutions','PERatio', 'ForwardPE','MarketCapitalization','AnalystTargetPrice','Industry','Sector']
    data_points_kmeans = data_points
    stock_returns = pd.DataFrame(columns=['ticker']+data_points)
    if doEarnings:
        stock_returns = pd.DataFrame(columns=['ticker']+data_points+earn_points)
        data_points_kmeans = data_points+earn_points
    runListA =  b.stock_list
    runList=[]
    allList = sqlcursor.cursor().execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
    for a in runListA:
        if [a[0]] not in runList:
            runList+=[[a[0]]]
    for a in allList:
        #if len(runList)>1000:
        #    break
        if [a[0]] not in runList:
            runList+=[[a[0]]]

    print('Running: ',len(runList))
    #for s in b.etfs:
    for s in runList:
        if s[0].count('^'):
            continue
        print(s[0])
        
        estock_info,j=ConfigTable(s[0], sqlcursor,ts,readType, j)
        if len(estock_info)==0 or len(estock_info)<181:
            continue
        #print(estock_info['adj_close'][-30:])
        sdf = pd.DataFrame([[s[0],(estock_info['close'].values[-1]-estock_info['open'].values[-1])/estock_info['open'].values[-1],
                               #estock_info['adj_close'].pct_change(1),
                               estock_info['adj_close'].pct_change(2).values[-1],
                               estock_info['adj_close'].pct_change(3).values[-1],
                               estock_info['adj_close'].pct_change(4).values[-1],
                               estock_info['adj_close'].pct_change(5).values[-1],
                               estock_info['adj_close'].pct_change(15).values[-1],
                               estock_info['adj_close'].pct_change(30).values[-1],
                               estock_info['adj_close'].pct_change(60).values[-1],
                               estock_info['adj_close'].pct_change(180).values[-1],
                               (estock_info['high'].values[-1]-estock_info['low'].values[-1])/estock_info['open'].values[-1],
                               estock_info['adj_close'][-5:].std(),
                               estock_info['adj_close'][-30:].std(),
                               estock_info['adj_close'][-180:].std()]],columns=['ticker']+data_points)

        # Load the earnings if requested
        if doEarnings:
            stockInfoQuarter,stockInfoAnnual,company_overview=CollectEarnings(s[0],connectionCal)
            if len(company_overview)==0:
                stock_returns=stock_returns.append(sdf,ignore_index=True)
                print(f'ERROR no overview {s[0]}')
                continue
            #print(stock_returns)

            company_overview_latest = company_overview[earn_points][-1:]
            company_overview_latest['Industry'].values[-1] = BuildIndustryMap(company_overview_latest['Industry'].values[-1],ind_map)
            company_overview_latest['Sector'].values[-1] = BuildIndustryMap(company_overview_latest['Sector'].values[-1],sec_map)
            company_overview_latest['AnalystTargetPrice'].values[-1] =  company_overview_latest['AnalystTargetPrice'].values[-1]/estock_info['adj_close'].values[-1]
            company_overview_latest['AnalystTargetPrice'].values[-1] =  company_overview_latest['AnalystTargetPrice'].values[-1]/estock_info['adj_close'].values[-1]
            for e in earn_points:
                sdf[e] = company_overview_latest[e].values
        sys.stdout.flush()
        
        stock_returns=stock_returns.append(sdf,ignore_index=True)

    if doEarnings:
        maxMarketCapitalization = stock_returns['MarketCapitalization'].max()
        stock_returns['MarketCapitalization'] /= maxMarketCapitalization
        print(maxMarketCapitalization)
    print(ind_map)
    print(sec_map)
    print(stock_returns)
    stock_returns.to_csv('stock_returns_kmeansv2.csv',index=False)
    #scaler = StandardScaler()
    #scaled_features = scaler.fit_transform(X)

    # A list holds the SSE values for each k
    sse = []
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    maxTestCluster = 31
    for k in range(2, maxTestCluster):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(stock_returns[data_points_kmeans].dropna())
        sse.append(kmeans.inertia_)

        #kmeans.fit(scaled_features)
        #score = silhouette_score(scaled_features, kmeans.labels_)
        #silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, maxTestCluster), sse)
    plt.xticks(range(2, maxTestCluster))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    #plt.style.use("fivethirtyeight")
    #plt.plot(range(2, 11), silhouette_coefficients)
    #plt.xticks(range(2, 11))
    #plt.xlabel("Number of Clusters")
    #plt.ylabel("Silhouette Coefficient")
    #plt.show()
    
