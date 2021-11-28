import os,sys
import datetime
import pandas as pd
import numpy as np
import base as b
import math
import datetime,time
from bs4 import BeautifulSoup
from ReadData import is_date,SQL_CURSOR,UpdateTable,GetTimeSlot

def LoadData(df, sqlcursorExtra, tableName='SPY200MA', index_label='Date'):

    todayDateTime=datetime.datetime.now()
    Load=True
    try:
        stock = pd.read_sql('SELECT * FROM %s' %tableName, sqlcursorExtra) #,index_col='Date')
        #print(stock.columns)
        stock['Date']=pd.to_datetime(stock.Date.astype(str), format='%Y-%m-%d')
        stock['Date']=pd.to_datetime(stock['Date'])
        stock = stock.set_index('Date')
        stock = stock.sort_index()

        if len(stock)>0 and (todayDateTime - stock.index[-1])<datetime.timedelta(days=1,hours=20):
            print('already loaded %s! %s' %(tableName,stock.index[-1]))
            Load=False
    except pd.io.sql.DatabaseError:
        pass

    if Load:
        UpdateTable(df,tableName,sqlcursorExtra, index_label=index_label)

# Collect info from finviz
def collect(sqlcursorExtra, URLin = 'https://finviz.com/screener.ashx?v=340\&s=ta_topgainers\&r=',tableName='',doHighLow=False):
    """collect - 
    input: URL - str - path
           """
    # get date
    yesterday = datetime.datetime.now() - datetime.timedelta(1)

    URL = URLin
    filename_rec='/tmp/topgain%s.html' %tableName
    os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))
    if doHighLow:
        table_MN = pd.read_html(filename_rec)
        if len(table_MN)==0:
            return
        table_MN = table_MN[0]
        col_names={}
        isUnnamed=False
        for c in table_MN.columns:
            col_names[c]=c.replace(' ','_').replace('>','gt').replace('<','lt')
            if c.count('Unnamed'): isUnnamed=True
        if isUnnamed:
            if len(table_MN.columns)!=11 and len(table_MN.columns)!=12:
                print(table_MN.columns)
            if len(table_MN.columns)==12:
                table_MN.columns = ['ticker','price','perc_change','dollar_change','rating','volume_times_price','volume','mkt_cap','p_to_e','eps','num_employees','sector']
                table_MN = table_MN[['ticker','price','perc_change','dollar_change','rating','volume','mkt_cap','p_to_e','eps','num_employees','sector']]
            else:
                table_MN.columns = ['ticker','price','perc_change','dollar_change','rating','volume','mkt_cap','p_to_e','eps','num_employees','sector']                
            for it in  ['price','perc_change','dollar_change','volume','mkt_cap','p_to_e','eps','num_employees']:
                table_MN[it].replace({'\%':'','M':'000000','K':'000','B':'000000000'},inplace=True,regex=True)
                table_MN[it] = pd.to_numeric(table_MN[it],errors='coerce')
        table_MN.rename(columns=col_names,inplace=True)
        table_MN['Date'] = datetime.datetime.strftime(yesterday, '%Y-%m-%d')
        table_MN['Date']=pd.to_datetime(table_MN['Date'])
        print(table_MN)
        LoadData(table_MN, sqlcursorExtra, tableName=tableName,index_label=None)
        return table_MN
    else:
        soup = BeautifulSoup(open(filename_rec,'r'), 'html.parser')
        td = soup.find_all('cheat-sheet')        
        outlist = {}
        for t in td:
            j=eval(t['data-cheat-sheet-data'])
            for g in j:
                #print(g['value'],g['labelSupportResistance'],g['labelTurningPoints'])
                if g['labelSupportResistance']==g['labelTurningPoints']:
                    try:
                        outlist[g['labelSupportResistance'].replace(' ','_').replace('%','perc')]=float(g['value'])
                    except ValueError:
                        outlist[g['labelSupportResistance'].replace(' ','_').replace('%','perc')]=None
                else:
                    try:
                        outlist[(g['labelSupportResistance']+g['labelTurningPoints']).replace(' ','_').replace('%','perc')]=float(g['value'])
                    except ValueError:
                        outlist[(g['labelSupportResistance']+g['labelTurningPoints']).replace(' ','_').replace('%','perc')]=None
        #print(outlist)
        df = pd.DataFrame([outlist])
        df['Date'] = datetime.datetime.strftime(yesterday, '%Y-%m-%d')
        df['Date']=pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        print(df)
        LoadData(df, sqlcursorExtra, tableName=tableName)
        return outlist

if __name__ == "__main__":
    # execute only if run as a script
    sqlcursor = SQL_CURSOR(db_name='stocksPerfHistory.db')

    #MMOH -> NYSE
    #200 dma S&P 500
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S5TH/cheat-sheet',tableName='SPY200MA')
    #50 dma S&P 500, S5FI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S5FI/cheat-sheet',tableName='SPY50MA')
    #100 dma S&P 500, S5OH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S5OH/cheat-sheet',tableName='SPY100MA')
    #20 dma S&P 500, S5TW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S5TW/cheat-sheet',tableName='SPY20MA')

    #200 dma S&P 100
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S1TH/cheat-sheet',tableName='SPYonehun200MA')
    #50 dma S&P 100, S1FI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S1FI/cheat-sheet',tableName='SPYonehun50MA')
    #100 dma S&P 100, S1OH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S1OH/cheat-sheet',tableName='SPYonehun100MA')
    #20 dma S&P 100, S1TW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$S1TW/cheat-sheet',tableName='SPYonehun20MA')

    #200 dma NASDAQ 100
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$NDTH/cheat-sheet',tableName='NASDAQonehun200MA')
    #50 dma NASDAQ 100, NDFI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$NDFI/cheat-sheet',tableName='NASDAQonehun50MA')
    #100 dma NASDAQ 100, NDOH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$NDOH/cheat-sheet',tableName='NASDAQonehun100MA')
    #20 dma NASDAQ 100, NDTW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$NDTW/cheat-sheet',tableName='NASDAQonehun20MA')

    #200 dma Russell 1000
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R1TH/cheat-sheet',tableName='R1k200MA')
    #50 dma Russell 1000, R1FI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R1FI/cheat-sheet',tableName='R1k50MA')
    #100 dma Russell 1000, R1OH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R1OH/cheat-sheet',tableName='R1k100MA')
    #20 dma Russell 1000, R1TW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R1TW/cheat-sheet',tableName='R1k20MA')

    #200 dma Russell 2000
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R2TH/cheat-sheet',tableName='R2k200MA')
    #50 dma Russell 2000, R2FI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R2FI/cheat-sheet',tableName='R2k50MA')
    #100 dma Russell 2000, R2OH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R2OH/cheat-sheet',tableName='R2k100MA')
    #20 dma Russell 2000, R2TW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R2TW/cheat-sheet',tableName='R2k20MA')

    #200 dma Russell 3000
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R3TH/cheat-sheet',tableName='R3k200MA')
    #50 dma Russell 3000, R3FI
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R3FI/cheat-sheet',tableName='R3k50MA')
    #100 dma Russell 3000, R3OH
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R3OH/cheat-sheet',tableName='R3k100MA')
    #20 dma Russell 3000, R3TW
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$R3TW/cheat-sheet',tableName='R3k20MA')
    
    #200 dma NYSE
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$MMTH/cheat-sheet',tableName='NYSE200MA')
    #50 dma NYSE
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$MMFI/cheat-sheet',tableName='NYSE50MA')
    #50 dma NYSE
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$MMFI/cheat-sheet',tableName='NYSE100MA')
    #50 dma NYSE
    total_table_top_gain = collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/quotes/\$MMTW/cheat-sheet',tableName='NYSE20MA')

    # collect the stats for the day
    collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/highs-lows/summary',tableName='summary',doHighLow=True)

    # potentially partial, but pretty good
    collect(sqlcursor, URLin = 'https://www.tradingview.com/markets/stocks-usa/market-movers-ath/',tableName='summaryat_high',doHighLow=True)
    collect(sqlcursor, URLin = 'https://www.tradingview.com/markets/stocks-usa/market-movers-atl/',tableName='summaryat_low',doHighLow=True)
    collect(sqlcursor, URLin = 'https://www.tradingview.com/markets/stocks-usa/market-movers-52wk-high/',tableName='summary52w_high',doHighLow=True)
    collect(sqlcursor, URLin = 'https://www.tradingview.com/markets/stocks-usa/market-movers-52wk-low/',tableName='summary52w_low',doHighLow=True)

    # struggled to load these
    #collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/highs-lows/highs?screener=overall\&timeFrame=alltime\&page=all',name='summary',doHighLow=True)
    #collect(sqlcursor, URLin = 'https://www.barchart.com/stocks/highs-lows/lows?screener=overall\&timeFrame=alltime\&page=all',name='summary',doHighLow=True)
    #

    # need something special to load this
    #collect(sqlcursor, URLin = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',name='fed_schedule',doHighLow=True)
    #earnings dates?
