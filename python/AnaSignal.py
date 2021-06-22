from dataclasses import dataclass
from ReadData import SQL_CURSOR,ALPHA_FundamentalData,ConfigTable,ALPHA_TIMESERIES,AddInfo,GetTimeSlot,ALPACA_REST,runTicker,getQuotesTS,getQuotes
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import numpy as np
from datetime import datetime
import datetime as maindatetime
import matplotlib.pyplot as plt
import statsmodels.api as sm1
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pmd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pytz,os
debug=False
draw=True

#code snippet 5.1
# Fit linear regression on close
# Return the t-statistic for a given parameter estimate.
def tValLinR(close,plot=True):
    #tValue from a linear trend
    x = np.ones((close.shape[0],2)) # adding the constant
    x[:,1] = np.arange(close.shape[0])
    #print(x,close)
    ols = sm1.OLS(close, x).fit()
    #print(ols)
    print(ols.summary())
    if plot:
        PlotFit(ols,close)
    time_today = np.datetime64(datetime.today())
    today_value_predicted = ols.predict(np.array((1,len(close)-1+(time_today-close.index.values[-1])/np.timedelta64(5724000, 's'))))
    prstd, iv_l, iv_u = wls_prediction_std(ols)
    #if today_value_predicted[0]>ols.fittedvalues[-1]:
    #    (ols.fittedvalues[-1] - today_value_predicted[0]) / (iv_u[-1] - ols.fittedvalues[-1])
    #if today_value_predicted[0]<ols.fittedvalues[-1]:
    #    (today_value_predicted[0] - ols.fittedvalues[-1]) / (ols.fittedvalues[-1] - iv_l[-1])
        #print(ols.predict(np.timedelta64(datetime.today())))
    #print(ols.params) # constant + slope
    #print(ols.tvalues)
    return ols.tvalues[1],today_value_predicted # grab the t-statistic for the slope

def PlotFit(res,x):

    prstd, iv_l, iv_u = wls_prediction_std(res)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, 'o', label='data')
    #ax.plot(x, y_true, 'b-', label="True")
    ax.plot(x.index, res.fittedvalues, 'r--.', label="OLS")
    ax.plot(x.index, iv_u, 'r--')
    ax.plot(x.index, iv_l, 'r--')

    # Draw the predictions for one year into the future
    time_today = np.datetime64(datetime.today())
    #print((time_today-x.index.values[-1])/np.timedelta64(5724000, 's'))
    dtRange = np.linspace(pd.Timestamp(time_today).value,pd.Timestamp(time_today+np.timedelta64(31536000, 's')).value, 4)
    dtRange = pd.to_datetime(dtRange)
    Xnew = np.ones((x.shape[0]+4,2)) # adding the constant
    Xnew[:,1] = np.arange(x.shape[0]+4) 
    Xnew = sm1.add_constant(Xnew)
    #print(Xnew)
    ynewpred =  res.predict(Xnew)
    #print(ynewpred)
    ax.plot(x.index.union(dtRange), ynewpred, 'r', label="OLS prediction")
    ax.legend(loc='best');
    if draw: plt.show()

def find_closest_date(timepoint, time_series, add_time_delta_column=True):
    # takes a pd.Timestamp() instance and a pd.Series with dates in it
    # calcs the delta between `timepoint` and each date in `time_series`
    # returns the closest date and optionally the number of days in its time delta
    deltas = np.abs(time_series - timepoint)
    idx_closest_date = np.argmin(deltas)
    res = {"closest_date": time_series.iloc[idx_closest_date]}
    idx = ['closest_date']
    if add_time_delta_column:
        res["closest_delta"] = deltas[idx_closest_date]
        idx.append('closest_delta')
    return pd.Series(res, index=idx)

@dataclass
class Earnings:
    """Store company earnings, analyze them

        Parameters:
             ticker:            str   = None # ticker symbol
             income_statement_quarterly: dateFrame of quarterly statements.  Most of the key data!!
             company_overview: dateFrame of company overview. only info from what I saved
             balance_sheet_annual: dateFrame of annual balance sheet. Annual data
             stockInfoQuarter: dateFrame of quarterly earnings. expected and reported EPS
             stockInfoAnnual: dateFrame of annual earnings. just the EPS
             tstock_info: dateFrame of stock price data
             minute_prices: dateFrame of stock price data by minute for the last 5 days
             recent_quotes: dateFrame of recent stock quotes
    """
    ticker:            str   = None # ticker symbol
    income_statement_quarterly: str = None
    company_overview: str = None
    balance_sheet_annual: str = None
    stockInfoQuarter: str = None
    stockInfoAnnual: str = None
    tstock_info: str = None
    minute_prices: str = None
    recent_quotes: str = None

    # Merge on the 
    def mergeEPS(self):
        self.stockInfoQuarter = self.stockInfoQuarter.merge(self.tstock_info, how='left', right_index=True, left_index=True)
        self.stockInfoQuarter.sort_index(ascending=True,inplace=True)
        return self.stockInfoQuarter
    def mergeBalance(self):
        # find the closest date in the open market to that last date in the quarter's balance sheet report
        balance_sheet_annual=self.balance_sheet_annual
        self.tstock_info['Datesort'] = self.tstock_info.index
        balance_sheet_annual[['closest', 'days_bt_x_and_y']] = balance_sheet_annual.fiscalDateEnding.apply(find_closest_date, args=[ self.tstock_info.Datesort])
        self.balance_sheet_annual = self.balance_sheet_annual.merge(tstock_info, how='left', left_on='closest',right_index=True) 
        return self.balance_sheet_annual

    def mergeQuarterIncome(self):
        # find the closest date in the open market to that last date in the quarter's income report
        income_statement_quarterly=self.income_statement_quarterly
        self.tstock_info['Datesort'] = self.tstock_info.index
        income_statement_quarterly[['closest', 'days_bt_x_and_y']] = income_statement_quarterly.fiscalDateEnding.apply(find_closest_date, args=[ self.tstock_info.Datesort])
        self.income_statement_quarterly = self.income_statement_quarterly.merge(self.tstock_info, how='left', left_on='closest', right_index=True)
        self.income_statement_quarterly.set_index('fiscalDateEnding',inplace=True)
        self.income_statement_quarterly.sort_index(ascending=True,inplace=True)
        return self.income_statement_quarterly
    
    # fit the revenue and ask what the surprise is from the most recent result
    def TrendingEPS(self):
        return tValLinR(self.stockInfoQuarter.reportedEPS)
    
    # fit the revenue and ask what the surprise is from the most recent result
    # costOfRevenue,costofGoodsAndServicesSold,totalRevenue,grossProfit
    # ebit = operating earnings
    # ebidta = Earnings Before Interest, Taxes, Depreciation, and Amortization 
    def TrendingRevenue(self,fitVar='totalRevenue'):
        return tValLinR(self.income_statement_quarterly[fitVar])
    
    # Fit with an OLS the value of the Price (centered 20d MA) over the total revenue
    def PriceOverTotalRev(self):
        self.income_statement_quarterly['p_over_rev'] =  self.income_statement_quarterly['sma20cen']/self.income_statement_quarterly['totalRevenue']
        return tValLinR(self.income_statement_quarterly['p_over_rev'],True)

    # get the fraction of the bolanger band
    def getPercentOfBBand(self):
        print(self.tstock_info.adj_close[-1])
        print(self.tstock_info.sma10[-1])
        print(self.tstock_info.BolUpper[-1])
        print(self.tstock_info.BolLower[-1])
        
    # get the fraction of the bolanger band
    def ARIMA(self,model_var='adj_close',n_periods=50,timescale='D'):
        #timeseries = self.tstock_info[model_var]
        timeseries=None

        if model_var=='close':
            timeseries = self.minute_prices[model_var]
            timeseries.index = np.arange(0,len(timeseries))
        else:
            timeseries = GetTimeSlot(self.tstock_info, days=5*365)[model_var]
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(311)
        fig = plot_acf(timeseries, ax=ax1, title="Autocorrelation on Original Series") 
        ax2 = fig.add_subplot(312)
        fig = plot_acf(timeseries.diff().dropna(), ax=ax2,  title="1st Order Differencing")
        ax3 = fig.add_subplot(313)
        fig = plot_acf(timeseries.diff().diff().dropna(), ax=ax3,  title="2nd Order Differencing")
        
        #model = ARIMA(timeseries, order=(1, 1, 1))
        #results = model.fit()
        #results.plot_predict(1, 210)
        autoarima_model = pmd.auto_arima(timeseries, start_p=1,start_q=1,test="adf",trace=True)
        #timeseries['ARIMA'] =
        fitted,confint = autoarima_model.predict(n_periods,return_conf_int=True,start=timeseries.index[-1])
        fittedv = autoarima_model.predict_in_sample()
        index_of_fc = pd.date_range(timeseries.index[-1], periods = n_periods, freq=timescale)
        if model_var=='close':
            index_of_fc = np.arange(timeseries.index[-1],+timeseries.index[-1]+n_periods)
        # make series for plotting purpose
        plt.show()
        fittedv_series = pd.Series(fittedv, index=timeseries.index)
        fitted_series = pd.Series(fitted, index=index_of_fc)
        print(fittedv_series - timeseries)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)
        print(lower_series)
        print(fitted_series)
        print(upper_series)
        # Plot
        plt.plot(timeseries)
        plt.plot(fitted_series, color='darkgreen')
        plt.plot(fittedv_series, color='yellow')
        plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

        plt.title("SARIMA - Final Forecast of Stock prices - Time Series Dataset")
        plt.show()

    # Build the recent returns
    def BuildPDF(self):
        minute_stats = self.minute_prices['open'].describe()
        #print(minute_stats)
        print(self.minute_prices['open'].mean())
        print(self.minute_prices['open'].std())
        print(self.minute_prices['open'].max())
        print(self.minute_prices['open'].min())
        print(self.minute_prices['open'].quantile(0.75))

        if len(self.tstock_info)>0:
            print('VWAP10: %s' %self.tstock_info['vwap10'][-1])
            print('Prev Close: %s' %self.tstock_info['close'][-1])
        if len(self.recent_quotes)>0:
            print(recent_quotes['bid_price'][-1])
        #print(self.minute_prices[''])
        #sys.exit(0)

    # Build the recent returns
    def WriteCSV(self, out_file_name = 'out_bull_instructions.csv',price_targets=[]):
        #ticker lot buy_limit target signal_date sold sold_at_loss ma200 ma50 ma20 ma10 vwap10 downSL upSL support_down BolUpper fivedaymax fivedaymin fivedaymean prev_close sell_date
        if len(self.tstock_info)>0:
            data_frame = self.tstock_info[['sma200', 'sma50', 'sma20', 'sma10', 'vwap10','BolUpper','BolLower','downSL','upSL']][-1:]
            minute_stats = self.minute_prices['open'].describe()
            data_frame['downSL'] = (1.0+data_frame['downSL'])*self.tstock_info['close'][-1]
            data_frame['upSL'] = (1.0+data_frame['upSL'])*self.tstock_info['close'][-1]
            data_frame['fivedaymax'] = self.minute_prices['open'].max()
            data_frame['fivedaymin'] = self.minute_prices['open'].min()
            data_frame['fivedaymean'] = self.minute_prices['open'].mean()
            est = pytz.timezone('US/Eastern')
            data_frame['signal_date'] = datetime.now(tz=est).strftime("%Y-%m-%dT%H:%M:%S-04:00")
            data_frame['sell_date'] = 'X'
            data_frame['sold_at_loss'] = 0
            data_frame['sold'] = 0
            data_frame['buy_limit'] = self.tstock_info['close'][-1]
            data_frame['prev_close'] = self.tstock_info['close'][-1]
            data_frame['lot'] = 5000
            if len(price_targets)>0:
                data_frame['target_after'] = price_targets[0]
                data_frame['target_before'] = 0.0
            if len(price_targets)>1:                
                data_frame['target_before'] = price_targets[1]
            data_frame['ticker'] = self.ticker
            #print(data_frame)
            
            os.system('cp Instructions/%s %s' %(out_file_name,out_file_name))
            dfnow=[]
            try:
                dfnow = pd.read_csv(out_file_name, sep=' ')
                if debug:
                    print(dfnow)
                    print(dfnow['ticker'])
                    print(dfnow[dfnow['ticker']==self.ticker]['sold'])
                    print(dfnow[dfnow['ticker']==self.ticker]['sold_at_loss'])
            except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError): 
                dfnow=[]
            os.system('rm %s' %out_file_name)
            if len(dfnow)==0:
                dfnow = data_frame
            elif self.ticker not in dfnow['ticker'].values:
                dfnow  = pd.concat([dfnow,data_frame])
            elif self.ticker in dfnow['ticker'].values:
                # make sure that this stock was already sold and was not sold at a loss. if so, then replace the current line
                if dfnow[dfnow['ticker']==self.ticker]['sold'].values[0]>0 and dfnow[dfnow['ticker']==self.ticker]['sold_at_loss'].values[0]==0:
                    print('sold')
                    dfnow[dfnow['ticker']==self.ticker] = data_frame[data_frame['ticker']==self.ticker]
                elif dfnow[dfnow['ticker']==self.ticker]['sold'].values[0]>0 and dfnow[dfnow['ticker']==self.ticker]['sold_at_loss'].values[0]>0:
                    #dfnow['sell_date']=pd.to_datetime(dfnow['sell_date'],errors='coerce')
                    #print(dfnow[dfnow['ticker']==self.ticker]['sell_date'].values[0])
                    time_of_sale = datetime.strptime(dfnow[dfnow['ticker']==self.ticker]['sell_date'].values[0],"%Y-%m-%dT%H:%M:%S-04:00")
                    time_of_sale = time_of_sale.replace(tzinfo=est)
                    # if more than 35 days, then let's remove it or replace it.
                    if time_of_sale<(datetime.now(tz=est)+maindatetime.timedelta(days=-35)):
                        print('old input')
                        dfnow[dfnow['ticker']==self.ticker] = data_frame[data_frame['ticker']==self.ticker]
            dfnow.to_csv(out_file_name, sep=' ',mode='a',index=False)
            os.system('cp %s Instructions/%s' %(out_file_name,out_file_name))
            
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

def GenerateSignal(ticker, out_file_name = 'out_bull_instructions.csv',price_targets=[]):
    connectionCal = SQL_CURSOR('earningsCalendarv2.db')
    fd = ALPHA_FundamentalData()
    sqlcursor = SQL_CURSOR()
    ts = ALPHA_TIMESERIES()
    api = ALPACA_REST()

    stockInfoQuarter,stockInfoAnnual,company_overview=CollectEarnings(ticker,connectionCal)

    # annual balance sheet
    balance_sheet_annual = fd.get_balance_sheet_annual(ticker)[0]
    balance_sheet_annual['fiscalDateEnding']=pd.to_datetime(balance_sheet_annual['fiscalDateEnding'],errors='coerce')
    for d in balance_sheet_annual.columns:
        if d not in ['fiscalDateEnding','totalAssets']:
            balance_sheet_annual[d]=pd.to_numeric(balance_sheet_annual[d],errors='coerce')
    # quarterly income statement
    income_statement_quarterly = fd.get_income_statement_quarterly(ticker)[0]
    for d in income_statement_quarterly.columns:
        if d not in ['fiscalDateEnding','reportedCurrency']:
            income_statement_quarterly[d]=pd.to_numeric(income_statement_quarterly[d],errors='coerce')
    for d in ['fiscalDateEnding']:
        income_statement_quarterly[d]=pd.to_datetime(income_statement_quarterly[d],errors='coerce')
    if debug:
        print(income_statement_quarterly)
        print(income_statement_quarterly.dtypes)
    
    tstock_info,j=ConfigTable(ticker, sqlcursor, ts,'compact')
    spy,j = ConfigTable('SPY', sqlcursor,ts,'compact')
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(tz=est) + maindatetime.timedelta(minutes=-40)
    #today = datetime.utcnow() + maindatetime.timedelta(minutes=-30)
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    five_days = (today + maindatetime.timedelta(days=-7)).strftime("%Y-%m-%dT%H:%M:%S-04:00")
    
    minute_prices  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=five_days, end=d1)
    # may want to restrict to NYSE open times
    try:
        AddInfo(spy, spy)
        AddInfo(tstock_info, spy, AddSupport=True)
    except (ValueError,KeyError):
        print('Error processing %s' %ticker)

    recent_quotes = getQuotes(api,ticker)
    if debug: print(tstock_info[['adj_close','sma20','sma20cen','vwap10cen','vwap10']][50:-10])
    earn = Earnings(ticker,income_statement_quarterly,company_overview,balance_sheet_annual,stockInfoQuarter,stockInfoAnnual,tstock_info,minute_prices,recent_quotes)
    #earn.BuildPDF()
    earn.WriteCSV(out_file_name,price_targets)

if __name__ == "__main__":
    # execute only if run as a script
    connectionCal = SQL_CURSOR('earningsCalendarv2.db')
    fd = ALPHA_FundamentalData()
    sqlcursor = SQL_CURSOR()
    ts = ALPHA_TIMESERIES()
    api = ALPACA_REST()
    ticker='MPC'
    ticker='RIOT'
    ticker='X'
    ticker='HZO'
    ticker='WOOF'
    #ticker='GOOGL'
    #ticker='F'
    stockInfoQuarter,stockInfoAnnual,company_overview=CollectEarnings(ticker,connectionCal)

    # annual balance sheet
    balance_sheet_annual = fd.get_balance_sheet_annual(ticker)[0]
    balance_sheet_annual['fiscalDateEnding']=pd.to_datetime(balance_sheet_annual['fiscalDateEnding'],errors='coerce')    
    for d in balance_sheet_annual.columns:
        if d not in ['fiscalDateEnding','totalAssets']:
            balance_sheet_annual[d]=pd.to_numeric(balance_sheet_annual[d],errors='coerce')
    if debug:
        print(balance_sheet_annual)
        print(balance_sheet_annual.dtypes)
    
    #company_overview = fd.get_company_overview(ticker)[0]
    #for d in company_overview.columns:
    #    if d not in ['Symbol','AssetType','Name','Description','CIK','Exchange','Currency','Country','Sector','Industry','Address','FiscalYearEnd','LatestQuarter','DividendDate','ExDividendDate','LastSplitFactor','LastSplitDate']:
    #        company_overview[d]=pd.to_numeric(company_overview[d],errors='coerce')
    #for d in ['LatestQuarter','DividendDate','ExDividendDate','LastSplitFactor','LastSplitDate']:
    #    company_overview[d]=pd.to_datetime(company_overview[d],errors='coerce')
    #if debug:
    #    print(company_overview)
    #    print(company_overview.dtypes)

    # quarterly income statement
    income_statement_quarterly = fd.get_income_statement_quarterly(ticker)[0]
    for d in income_statement_quarterly.columns:
        if d not in ['fiscalDateEnding','reportedCurrency']:
            income_statement_quarterly[d]=pd.to_numeric(income_statement_quarterly[d],errors='coerce')
    for d in ['fiscalDateEnding']:
        income_statement_quarterly[d]=pd.to_datetime(income_statement_quarterly[d],errors='coerce')
    if debug:
        print(income_statement_quarterly)
        print(income_statement_quarterly.dtypes)
    
    tstock_info,j=ConfigTable(ticker, sqlcursor, ts,'compact')
    spy,j = ConfigTable('SPY', sqlcursor,ts,'compact')
    
    est = pytz.timezone('US/Eastern')
    today = datetime.now(tz=est) + maindatetime.timedelta(minutes=-40)
    #today = datetime.utcnow() + maindatetime.timedelta(minutes=-30)
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    five_days = (today + maindatetime.timedelta(days=-7)).strftime("%Y-%m-%dT%H:%M:%S-04:00")
    
    minute_prices  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=five_days, end=d1)
    # may want to restrict to NYSE open times
    #minute_prices = minute_prices[(minute_prices.index.hour>13) & (minute_prices.index.hour<20)
    #print(minute_prices)
    try:
        AddInfo(spy, spy)
        AddInfo(tstock_info, spy, AddSupport=True)
    except (ValueError,KeyError):
        print('Error processing %s' %ticker)

    recent_quotes = getQuotes(api,ticker)
    if debug: print(tstock_info[['adj_close','sma20','sma20cen','vwap10cen','vwap10']][50:-10])
    earn = Earnings(ticker,income_statement_quarterly,company_overview,balance_sheet_annual,stockInfoQuarter,stockInfoAnnual,tstock_info,minute_prices,recent_quotes)
    earn.BuildPDF()
    earn.WriteCSV()
    #print(getQuotesTS(ts,ticker))
    
    earn.mergeQuarterIncome()
    earn.mergeEPS()
    earn.mergeBalance()
    t_hat_slope_of_rev,predict_total_earnings_today = earn.TrendingRevenue('totalRevenue')
    #for e in ['costOfRevenue','costofGoodsAndServicesSold','grossProfit','ebit']: #totalRevenue
    #    earn.TrendingRevenue(e)
        
    earn.TrendingEPS()
    #if debug:
    #    print(earn.mergeQuarterIncome()[['adj_close','sma20','fiscalDateEnding','grossProfit','closest']])
    #    print(earn.mergeEPS()[['adj_close','sma20','reportedEPS']])
    #    print(earn.mergeBalance()[['adj_close','sma20','closest','fiscalDateEnding']])
    print(earn.tstock_info['adj_close'].values[-1]/predict_total_earnings_today[0])
    earn.PriceOverTotalRev()
    earn.getPercentOfBBand()
    earn.ARIMA()
    earn.ARIMA(model_var='close',n_periods=500,timescale='1min')
    #earn.TrendingProfit()
