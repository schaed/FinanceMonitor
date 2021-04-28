import os,sys,time,datetime,copy
from ReadData import SQL_CURSOR,GetUpcomingEarnings,AddInfo,ALPHA_TIMESERIES,ConfigTable,ALPHA_FundamentalData,GetTimeSlot
import pandas  as pd
import numpy as np
import base as b
import numpy as np
from techindicators import techindicators
ReDownload = False
readType='full'
debug=False
draw=False
doPDFs = False
doPlot = False
outdir = b.outdir
import matplotlib.pyplot as plt
import matplotlib
if not draw:
    matplotlib.use('Agg')

def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    # plotting
    plt.clf()
    plt.scatter(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()

def MakePlotMulti(xaxis, yaxis=[], colors=[], labels=[], xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    # plotting
    j=0
    plt.clf()
    for y in yaxis:
        plt.plot(xaxis,y,color=colors[j],label=labels[j])
        j+=1
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    plt.legend(loc="upper left")
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw: plt.show()
    if doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()

def ProcessTicker(ticker, earningsExp, sqlcursor,spy,j,connectionCal):
    print(ticker)
    if debug: print(earningsExp)
    tstock_info,j=ConfigTable(ticker, sqlcursor, ts, readType, j, hoursdelay=23)
    AddInfo(tstock_info,spy,debug=debug)
    prev_earnings = None
    overview = None
    try:
        overview = pd.read_sql('SELECT * FROM overview WHERE Symbol="%s"' %(ticker), connectionCal)
        prev_earnings = pd.read_sql('SELECT * FROM quarterlyEarnings WHERE ticker="%s"' %(ticker), connectionCal)
    except:
        print('no previous info for %s' %ticker)
        pass
    if debug: print(prev_earnings)
    prev_earnings['earningDiff'] = prev_earnings['reportedEPS'] - prev_earnings['estimatedEPS']

    # shift some of the inputs by one day to have the day before inputs
    #print(list(tstock_info.columns))
    for a in ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividendamt', 'splitcoef', 'pos_volume', 'neg_volume', 'sma10', 'sma20', 'sma50', 'sma100', 'sma200', 'rstd10', 'rsi10', 'cmf', 'BolLower', 'BolCenter', 'BolUpper', 'KeltLower', 'KeltCenter', 'KeltUpper', 'copp','daily_return_stddev14', 'beta', 'alpha', 'rsquare', 'sharpe', 'cci', 'stochK', 'stochD', 'obv', 'force', 'macd', 'macdsignal', 'bop', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_TRENDMODE', 'HT_SINE', 'HT_SINElead', 'HT_PHASORphase', 'HT_PHASORquad', 'adx', 'willr', 'ultosc', 'aroonUp', 'aroonDown', 'aroon', 'senkou_spna_A', 'senkou_spna_B', 'chikou_span', 'SAR', 'vwap14', 'vwap10', 'vwap20', 'chosc', 'market', 'corr14']:
        tstock_info[a+'_daybefore'] = tstock_info[a].shift(1)
    #print(tstock_info[['rsi10_daybefore','rsi10','willr_daybefore','willr']].tail())

    # cleaning the previous earnings data types
    prev_earnings['reportedDate'] = pd.to_datetime(prev_earnings['reportedDate'],errors='coerce')
    prev_earnings['fiscalDateEnding'] = pd.to_datetime(prev_earnings['fiscalDateEnding'],errors='coerce')
    if debug: print(tstock_info[['daily_return','adj_close','thrday_future_return','oneday_future_return']])
    # Collecting support levels
    prev_earnings['tech_levels'] = ''
    for earn_date in prev_earnings.fiscalDateEnding.values:
        # coverting this datetime64 to datetime
        earn_dateA = (earn_date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        earn_dateA=datetime.datetime.utcfromtimestamp(earn_dateA)
        prior_year_tstock_info = GetTimeSlot(tstock_info,startDate=earn_dateA)
        tech_levels = techindicators.supportLevels(prior_year_tstock_info,drawhlines=False)
        if len(tech_levels)>0:
            tech_levels = [str(level[1]) for level in tech_levels]
            prev_earnings.loc[prev_earnings.fiscalDateEnding==earn_date,['tech_levels']] = ','.join(tech_levels)
    
    # merging or joining on the report date
    merged_stock_earn = pd.merge(prev_earnings,tstock_info,how="left",left_on='reportedDate',right_index=True)
    merged_stock_earn['high_from_open'] = merged_stock_earn['high'] - merged_stock_earn['open']
    #merged_stock_earn['e_over_p_diff'] = merged_stock_earn['reportedEPS'].diff()/merged_stock_earn['adj_close'].diff()
    merged_stock_earn['e_over_p_diff'] = (merged_stock_earn['reportedEPS']/merged_stock_earn['adj_close']).diff(periods=-1)
    merged_stock_earn['e_over_p_test'] = (merged_stock_earn['reportedEPS']/merged_stock_earn['adj_close'])
    if debug: print(merged_stock_earn[['fiscalDateEnding','adj_close','e_over_p_test','e_over_p_diff','open']])

    # Drawing output
    if doPlot:
        for j in ['daily_return','oneday_future_return','thrday_future_return','high_from_open']:
            MakePlot( merged_stock_earn['earningDiff'],merged_stock_earn[j], xname='EarningsDiff',yname=j,saveName='earningDiff', hlines=[],title='earningDiff')
            MakePlot( merged_stock_earn['e_over_p_diff'],merged_stock_earn[j], xname='e_over_p_diff',yname=j,saveName='e_over_p_diff', hlines=[],title='e_over_p_diff')
    return merged_stock_earn
    
# collecting spy
sqlcursor = SQL_CURSOR()
ts = ALPHA_TIMESERIES()
j=0

# reading in the spy data
spy,j = ConfigTable('SPY', sqlcursor,ts,readType,hoursdelay=2)
AddInfo(spy,spy,debug=debug)

# processing new earnings
connectionCal = SQL_CURSOR('earningsCalendar.db')
connectionCalv2 = SQL_CURSOR('earningsCalendarForTraining.db')
fd = ALPHA_FundamentalData()
my_3month_calendar=GetUpcomingEarnings(fd,ReDownload)
print(my_3month_calendar)
it=0
all_merged_stock_earnings=[]
for it in range(0,len(my_3month_calendar)):
    ticker = my_3month_calendar['symbol'].values[it]
    merged_stock_earnings = ProcessTicker(ticker, my_3month_calendar[it:it+1],sqlcursor,spy,j,connectionCal)


    # merge these data frames
    if len(all_merged_stock_earnings)==0:
        all_merged_stock_earnings = merged_stock_earnings
    else:
        all_merged_stock_earnings = pd.concat([merged_stock_earnings,all_merged_stock_earnings])
    it+=1
    print(all_merged_stock_earnings)

    # write the output
    if it%20==0 and it!=0:
        #UpdateTable(stock, ticker, sqlcursor, index_label='Date')
        all_merged_stock_earnings.to_sql(ticker, connectionCalv2,if_exists='replace', index=True)
        #break
