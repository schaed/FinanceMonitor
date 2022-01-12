from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker,SQL_CURSOR,ConfigTable,GetTimeSlot,AddInfo,IS_ALPHA_PREMIUM_WAIT_ITER
import pandas as pd
import numpy as np
import sys
import datetime
import base as b
import pickle
import time
import os
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats.mstats import gmean
matplotlib.use('Agg') 
draw=False
doPDFs=False
outdir='/tmp/'

def MakePlotMulti(xaxis, yaxis=[], colors=[], labels=[], xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None):
    """ Generic plotting option for multiple plots 
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         xname : str
            x-axis name
         colors : array of strings
            colors compatible with matplotlib
         labels : array of str
            legend names
         yname : str
            y-axis name
         saveName : str
            Saved file name
         hlines : array of horizontal lines drawn in matplotlib
         title : str
            Title of plot    
         doSupport : bool
            Request generation of support lines on the fly
         my_stock_info : pandas data frame of stock timing and adj_close
    """
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

def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None, doScatter=False,doBox=False):
    """ Generic plotting with option to show support lines
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         xname : str
            x-axis name
         yname : str
            y-axis name
         saveName : str
            Saved file name
         hlines : array of horizontal lines drawn in matplotlib
         title : str
            Title of plot
         doSupport : bool
            Request generation of support lines on the fly
         my_stock_info : pandas data frame of stock timing and adj_close
         doScatter : bool - draw a scatter plot
         doBox : bool - draw a box plot for unique x-values
     """
    # plotting
    plt.clf()
    ax7=None
    fig7=None
    if doScatter:
        plt.scatter(xaxis,yaxis)
    elif doBox: 
        fig7, ax7 = plt.subplots()
        d1=[]
        for m in np.unique(xaxis.values):
            d1+=[yaxis.loc[xaxis==m].dropna()]
        bp = ax7.boxplot(d1,whis=[5,95],showmeans=True,notch=True)
        ax7.grid(True)
        ax7.legend([bp['medians'][0], bp['means'][0]],['median','mean'],loc="upper left")
        plt.title(saveName.replace('_',' '))
    else:
        plt.plot(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw and ax7!=None: fig7.show()
    elif draw: plt.show()
    if doPDFs and ax7!=None: fig7.savefig(outdir+'%s.pdf' %(saveName))
    elif doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    if ax7!=None: fig7.savefig(outdir+'%s.png' %(saveName))
    else: plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()


api = ALPACA_REST()
ts = ALPHA_TIMESERIES()
ticker='SPY'
#ticker='TLT'
#ticker='QQQ'
#ticker='GLD'
#ticker='HAL'
#ticker='GUSH'
ticker='AVCT'
spy = runTicker(api,ticker)
stock_info=None
spy=None
sqlcursor = SQL_CURSOR()
readType='full'

spy,j = ConfigTable(ticker, sqlcursor,ts,readType)
print('spy')
print(spy)

# add info
if len(spy)==0:
    print('ERROR - empy info %s' %ticker)
    
spy['daily_return']=spy['adj_close'].pct_change(periods=1)+1
spy['openclosepct'] = (spy.close-spy.open)/spy.open+1
spy['closeopenpct'] = (spy.open-spy.shift(1).close)/spy.shift(1).close+1
spy['afterhourspct'] = (spy.shift(-1).open-spy.close)/spy.close+1
spy['year']=spy.index.year
spy['day']=spy.index.day
spy['month']=spy.index.month
spy['dayofyear']=spy.index.dayofyear
spy['dayofweek']=spy.index.dayofweek
spy['weekofyear']=spy.index.isocalendar().week
spy['is_month_end']=spy.index.is_month_end
print(spy[['open','close','daily_return','adj_close','openclosepct','closeopenpct']])

# compute monthly returns
spy_month_grouped = spy.groupby(['year','month'])
end_of_month_idx = spy_month_grouped.day.transform(max) == spy['day']
spy_end_of_month = spy[end_of_month_idx]
MakePlotMulti(spy_end_of_month.index, yaxis=[(spy_month_grouped['openclosepct']).apply(gmean)-1,(spy_month_grouped['closeopenpct']).apply(gmean)-1,(spy_month_grouped['daily_return']).apply(gmean)-1], colors=['black','green','red'], labels=['trading hours','overnight','close to close'], xname='Date',yname='Returns',saveName=ticker+'_returns_daynight_monthly', hlines=[],title='',doSupport=False,my_stock_info=None)

spy_end_of_month['monthly_return'] = spy_end_of_month.adj_close.pct_change(periods=1)
spy_end_of_month_avg = spy_end_of_month.groupby('month').mean()
spy_end_of_month_std = spy_end_of_month.groupby('month').std()
spy_end_of_month_avg['signif'] = spy_end_of_month_avg.monthly_return / spy_end_of_month_std.monthly_return
MakePlot(spy_end_of_month_avg.index, spy_end_of_month_avg.monthly_return, xname='Month',yname='Avg. Monthly Returns',saveName=ticker+'_avg_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_end_of_month_std.index, spy_end_of_month_std.monthly_return, xname='Month',yname='Std Dev. Monthly Returns',saveName=ticker+'_stddev_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_end_of_month_avg.index, spy_end_of_month_avg.signif, xname='Month',yname='Signif. Monthly Returns',saveName=ticker+'_signif_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)

# scatter plot
MakePlot(spy_end_of_month.month, spy_end_of_month.monthly_return, xname='Month',yname='Monthly Returns',saveName=ticker+'_scatter_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None,doScatter=True)
MakePlot(spy_end_of_month.month, spy_end_of_month.monthly_return, xname='Month',yname='Monthly Returns',saveName=ticker+'_box_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None,doBox=True)

## remove unity
spy['daily_return'] -=1
spy['openclosepct'] -=1
spy['closeopenpct'] -=1

# Compare returns
print(spy[['openclosepct','daily_return','closeopenpct']].describe())
print(spy[['openclosepct','daily_return','closeopenpct']].corr())
#MakePlot(spy_end_of_month_avg.index, spy_end_of_month_avg.monthly_return, xname='Month',yname='Avg. Monthly Returns',saveName=ticker+'_avg_monthly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)

# compute weekly returns
spy_yearly_grouped = spy.groupby(['year','weekofyear'])
end_of_week_idx = spy_yearly_grouped.dayofyear.transform(max)==spy.dayofyear
spy_end_of_week = spy.loc[end_of_week_idx,:]
#spy_end_of_week.loc[end_of_week_idx,'weekly_return'] = spy_end_of_week.adj_close.pct_change(periods=1)
spy_end_of_week.loc[:,'weekly_return'] = spy_end_of_week['adj_close'].pct_change(periods=1)
#spy_end_of_week.loc[end_of_week_idx,['weekly_return']] = spy_end_of_week.adj_close.pct_change(periods=1)
spy_end_of_week_avg = spy_end_of_week.groupby('weekofyear').mean()
spy_end_of_week_std = spy_end_of_week.groupby('weekofyear').std()
spy_end_of_week_avg['signif'] = spy_end_of_week_avg.weekly_return / spy_end_of_week_std.weekly_return

#print(spy_end_of_week_avg)
MakePlot(spy_end_of_week_avg.index, spy_end_of_week_avg.weekly_return, xname='Week of year',yname='Avg. Weekly Returns',saveName=ticker+'_avg_weekly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_end_of_week_std.index, spy_end_of_week_std.weekly_return, xname='Week of year',yname='Std Dev. Weekly Returns',saveName=ticker+'_stddev_weekly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_end_of_week_avg.index, spy_end_of_week_avg.signif, xname='Week of year',yname='Signif. Weekly Returns',saveName=ticker+'_signif_weekly_returns', hlines=[],title='',doSupport=False,my_stock_info=None)

# compute day of week return.
spy_day_of_week_avg = spy.groupby('dayofweek').mean()
spy_day_of_week_std = spy.groupby('dayofweek').std()
spy_day_of_week_avg['signif'] = spy_day_of_week_avg.daily_return / spy_day_of_week_std.daily_return
MakePlot(spy_day_of_week_avg.index, spy_day_of_week_avg.daily_return, xname='Day of week',yname='Avg. Daily Returns',saveName=ticker+'_avg_daily_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_day_of_week_std.index, spy_day_of_week_std.daily_return, xname='Day of week',yname='Std Dev. Daily Returns',saveName=ticker+'_stddev_daily_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_day_of_week_std.index, spy_day_of_week_avg.signif, xname='Day of week',yname='Signif. Daily Returns',saveName=ticker+'_signif_daily_returns', hlines=[],title='',doSupport=False,my_stock_info=None)

# compute day of month return.
spy_day_of_month_avg = spy.groupby('day').mean()
spy_day_of_month_std = spy.groupby('day').std()
spy_day_of_month_avg['signif'] = spy_day_of_month_avg.daily_return / spy_day_of_month_std.daily_return
#print(spy_day_of_month_avg)
MakePlot(spy_day_of_month_avg.index, spy_day_of_month_avg.daily_return, xname='Day of month',yname='Avg. Daily Returns',saveName=ticker+'_avg_day_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_day_of_month_std.index, spy_day_of_month_std.daily_return, xname='Day of month',yname='Std Dev. Daily Returns',saveName=ticker+'_stddev_day_returns', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlot(spy_day_of_month_avg.index, spy_day_of_month_avg['signif'], xname='Day of month',yname='Signif. Daily Returns',saveName=ticker+'_signif_day_returns', hlines=[],title='',doSupport=False,my_stock_info=None)

# overnight vs daily
spy['openclosepctma20'] = ((spy['openclosepct']+1).rolling(20).apply(gmean)-1)
spy['closeopenpctma20'] = ((spy['closeopenpct']+1).rolling(20).apply(gmean)-1)
spy['daily_returnma20'] = ((spy['daily_return']+1).rolling(20).apply(gmean)-1)
spy['openclosepctmavg20'] = spy['openclosepct'].rolling(20).mean()
spy['closeopenpctmavg20'] = spy['closeopenpct'].rolling(20).mean()
spy['daily_returnmavg20'] = spy['daily_return'].rolling(20).mean()
spy['openclosepctma5'] = ((spy['openclosepct']+1).rolling(5).apply(gmean)-1)
spy['closeopenpctma5'] = ((spy['closeopenpct']+1).rolling(5).apply(gmean)-1)
spy['daily_returnma5'] = ((spy['daily_return']+1).rolling(5).apply(gmean)-1)
spy['openclosepctmag5'] = spy['openclosepct'].rolling(5).mean()
spy['closeopenpctmag5'] = spy['closeopenpct'].rolling(5).mean()
spy['daily_returnmag5'] = spy['daily_return'].rolling(5).mean()

spy_1y = GetTimeSlot(spy, days=365)
MakePlotMulti(spy.index, yaxis=[spy['openclosepctma20'],spy['closeopenpctma20'],spy['daily_returnma20']], colors=['black','green','red'], labels=['trading hours','overnight','close to close'], xname='Date',yname='Returns Geo MA20',saveName=ticker+'_returns_daynight', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlotMulti(spy_1y.index, yaxis=[spy_1y['openclosepctma20'],spy_1y['closeopenpctma20'],spy_1y['daily_returnma20']], colors=['black','green','red'], labels=['trading hours','overnight','close to close'], xname='Date',yname='Returns Geo MA20',saveName=ticker+'_returns_daynight_oneyear', hlines=[],title='',doSupport=False,my_stock_info=None)
MakePlotMulti(spy_1y.index, yaxis=[spy_1y['openclosepctma5'],spy_1y['closeopenpctma5'],spy_1y['daily_returnma5']], colors=['black','green','red'], labels=['trading hours','overnight','close to close'], xname='Date',yname='Returns Geo MA5',saveName=ticker+'_returns_daynight_oneyear_ma5', hlines=[],title='',doSupport=False,my_stock_info=None)

MakePlotMulti(spy_1y.index, yaxis=[spy_1y['openclosepct'],spy_1y['closeopenpct'],spy_1y['daily_return']], colors=['black','green','red'], labels=['trading hours','overnight','close to close'], xname='Date',yname='Returns',saveName=ticker+'_returns_daynight_oneyear_noMA', hlines=[],title='',doSupport=False,my_stock_info=None)

print(spy[['openclosepct','closeopenpct','open','close']])
#print(spy_end_of_month)
#print(spy_end_of_week)


# Compare returns
print(spy_1y[['openclosepct','daily_return','closeopenpct','afterhourspct']].describe())
print(spy_1y[['openclosepct','daily_return','closeopenpct','afterhourspct']].corr())
MakePlot(spy_1y['openclosepct'], spy_1y['afterhourspct'], xname='During trading hours',yname='After hours night after',saveName=ticker+'_trading_vs_nightafter', hlines=[],title='',doSupport=False,my_stock_info=None,doScatter=True)
MakePlot(spy_1y['openclosepct'], spy_1y['closeopenpct'], xname='During trading hours',yname='After hours night before',saveName=ticker+'_trading_vs_nightbefore', hlines=[],title='',doSupport=False,my_stock_info=None,doScatter=True)
MakePlot(spy_1y['openclosepct'], spy_1y['daily_return'], xname='During trading hours',yname='Returns',saveName=ticker+'_trading_vs_returns', hlines=[],title='',doSupport=False,my_stock_info=None,doScatter=True)
