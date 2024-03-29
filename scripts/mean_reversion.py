# Add spacy word analyzer
import spacy
from spacy.tokens import Token
nlp = spacy.load('en_core_web_sm')
from Sentiment import Sentiment,News
import glob,os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ReadData import ALPACA_REST,runTicker,ConfigTable,ALPHA_TIMESERIES,GetTimeSlot,SQL_CURSOR
from alpaca_trade_api.rest import TimeFrame
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pmd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
s=Sentiment()
debug=False
# create sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import statsmodels.api as sm1
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.dates as mdates
draw=False
debug=False
doPDFs=True
outdir='/tmp/'
api = ALPACA_REST()

inputTxt='Honest Company reports Q1 EPS (13c) vs. 1c last year'
#inputTxt='Lennar reports Q2 adjusted EPS $2.95, consensus $2.36'
#inputTxt='Cognyte reports Q1 EPS (20c), consensus (15c)'
#inputTxt='Brookdale Senior Living resumed with a Buy at Stifel'
#inputTxt='Anglo American price target raised to 3,670 GBp from 3,500 GBp at Morgan Stanley'
#inputTxt='GMS Inc. reports Q4 adjusted EPS $1.07, consensus 82c'
##inputTxt='CalAmp reports Q1 adjusted EPS 8c, consensus 7c'
##inputTxt='Adagene initiated with a Buy at China Renaissance'
inputTxt='Molecular Partners indicated to open at $20, IPO priced at $21.25'
#inputTxt='WalkMe indicated to open at $33.20, IPO priced at $31'
#inputTxt='Bassett Furniture reports Q2 EPS 60c, two est. 35c'
#inputTxt='Royal Caribbean reports Q2 adjusted EPS ($5.06), consensus ($4.40)'
#inputTxt='UroGen Pharma reports Q2 EPS ($1.17), consensus ($1.27)'

s.Parse(inputTxt,'Honest Company', 'HON', sid=sid, nlp=nlp, is_earnings=False)
#s.Sentiment(sid=sid,nlp=nlp,is_earnings=is_earnings)
print(s)

#sys.exit(0)
import pytz
import datetime
est = pytz.timezone('US/Eastern')

# add data for new fits
def AddData(t):
    
    t['slope']=0.0
    #t['slope'] = t.close.rolling(8).apply(tValLinR) 
    #t['slope_volume'] = t.volume.rolling(8).apply(tValLinR) 
    t['signif_volume'] = (t.volume-t.volume.mean())/t.volume.std()
    t['volma20'] = t.volume.rolling(20).mean()
    t['volma10'] = t.volume.rolling(10).mean()
    t['ma50'] = t.close.rolling(50).mean()
    fivma = t['ma50'].mean()
    t['ma50_div'] = (t['ma50'] - fivma)/fivma
    t['ma50_div'] *=10.0
    t['minute_diff_now'] = (t['close']-t['open'])/t['open']
    t['minute_diff_minago'] = t['minute_diff_now'].shift(1)
    t['minute_diff_2minago'] = t['minute_diff_now'].shift(2)
    t['minute_diff_15minago'] = t['minute_diff_now'].shift(15)
    t['minute_diff_5minback'] = (t['close'].shift(5)-t['open'].shift(1))/t['open'].shift(1)
    t['minute_diff_15minback'] = (t['close'].shift(15)-t['open'].shift(1))/t['open'].shift(1)
    t['minute_diff_15minfuture'] = (t['close'].shift(-15)-t['open'].shift(-1))/t['open'].shift(-1)
    t['minute_diff_5minfuture'] = (t['close'].shift(-5)-t['open'].shift(-1))/t['open'].shift(-1)
    t['minute_diff'] = (t['close']-t['open'])/t['open']
    #t['minute_diff'] = (t['close']-t['open'])/t['open']
    t['minute_diff'] *=10.0
    t['minute_diff']+=1.5
    
    t['minute_diffHL'] = (t['high']-t['low'])/t['open']
    t['minute_diffHL'] *=10.0
    t['minute_diffHL']+=1.75
    t['change'] = t.close
    t['change'] -= t.open.values[-1]
    t['change'] /= t.open.values[-1]
    
# add data for new fits
def AddDataShort(t):

    t['slope'] = t.close.rolling(8).apply(tValLinR) 
    t['slope_volume'] = t.volume.rolling(8).apply(tValLinR)
    t['time']  = t.index
    t['i'] = range(0,len(t))
    t.set_index('i',inplace=True)
    print(t[t.volume<500])
    max_vol = t.volume.max()
    t.volume/=max_vol
    m = t.open.mean()
    t['norm_open']  = t.open
    t.norm_open-=t.open.mean()
    t.norm_open/=m
    t.norm_open*=10.0
    t['norm_close']  = t.close
    t.norm_close-=m
    t.norm_close/=m
    t.norm_close*=10.0
    t['norm_high']  = t.high
    t.norm_high-=m
    t.norm_high/=m
    t.norm_high*=10.0
    t['norm_low']  = t.low
    t.norm_low-=m
    t.norm_low/=m
    t.norm_low*=10.0
    
# Fit linear regression on close
# Return the t-statistic for a given parameter estimate.
def tValLinR(close,plot=False,extra_vars=[],debug=False):
    #tValue from a linear trend
    x = np.ones((close.shape[0],2)) # adding the constant
    x[:,1] = np.arange(close.shape[0])
    if len(extra_vars)>0:
        x = np.ones((close.shape[0],3)) # adding the constant
        x[:,1] = np.arange(close.shape[0])    
        x[:,2] = extra_vars.volume#np.arange(close.shape[0])    
    #print(x)
    #print(x,close)
    ols = sm1.OLS(close, x).fit()
    #print(ols)
    if debug: print(ols.summary())
    #time_today = np.datetime64(datetime.today())
    #today_value_predicted = ols.predict(np.array((1,len(close)-1+(time_today-close.index.values[-1])/np.timedelta64(5724000, 's'))))
    prstd, iv_l, iv_u = wls_prediction_std(ols)
    return ols.tvalues[1] #,today_value_predicted # grab the t-statistic for the slope

# Plot the auto correlation functions
def ACF(timeseries,lags=50, arima_order=(9, 1, 1), sarima_order=None,n_periods=50,timescale='D',trend='n'): #seasonal_order
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(311)
    fig = plot_acf(timeseries, ax=ax1, title="Autocorrelation on Original Series") 
    ax2 = fig.add_subplot(312)
    fig = plot_acf(timeseries.diff().dropna(), ax=ax2,  title="1st Order Differencing")
    ax3 = fig.add_subplot(313)
    fig = plot_acf(timeseries.diff().diff().dropna(), ax=ax3,  title="2nd Order Differencing")
    plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(311)
    fig = plot_pacf(timeseries, ax=ax1, lags=lags, title="Partial Autocorrelation on Original Series") 
    ax2 = fig.add_subplot(312)
    fig = plot_pacf(timeseries.diff().dropna(), ax=ax2, lags=lags,  title="1st Order Differencing")
    ax3 = fig.add_subplot(313)
    fig = plot_pacf(timeseries.diff().diff().dropna(), ax=ax3, lags=lags,  title="2nd Order Differencing")
    plt.show()
    #plot_pacf(timeseries, lags=lags)
    #plt.show()
    model=None;
    if sarima_order is not None:
        model = ARIMA(timeseries, seasonal_order=sarima_order,trend=trend) # (p,d,q)        
    else:
        model = ARIMA(timeseries, order=arima_order,trend=trend) # (p,d,q)
    results = model.fit()
    print(results.summary())
    #print(results.aic)
    #print(results.fittedvalues)
    #print(results.sse)
    #print(results.get_forecast(50).predicted_mean) # predicted mean
    #print(results.get_forecast(50).se_mean) # standard error on the predicted mean
    #print(results.get_forecast(50).tvalues) #The ratio of the predicted mean to its standard deviation
    #print(results.get_forecast(50).var_pred_mean) #The variance of the predicted mean
    #print(results.get_forecast(50).conf_int()) #Confidence interval construction for the predicted mean.
    forecast = results.get_forecast(n_periods).summary_frame() #Summary frame of mean, variance and confidence interval.
    #plt.show()
    index_of_fc = forecast.index
    if timescale in ['D']:    
        index_of_fc = pd.date_range(timeseries.index[-1], periods = n_periods, freq=timescale)
        
    plt.plot(timeseries)
    plt.plot(results.fittedvalues[1:], color='darkgreen')
    plt.plot(index_of_fc,forecast['mean'].values, color='yellow')
    plt.fill_between(index_of_fc,#forecast.index, 
                    forecast['mean_ci_lower'], 
                    forecast['mean_ci_upper'], 
                    color='k', alpha=.15)

    plt.title("SARIMA - Final Forecast of Stock prices - Time Series Dataset")
    plt.show()
    
# get the fraction of the bolanger band
def ARIMAauto(timeseries,extra_vars=[],model_var='close',n_periods=50,fit_range_max=-1, verbose=False,timescale='D',seasonal_m=1):

    # test seasonal
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(311)
    fig = plot_acf(timeseries, ax=ax1, title="Autocorrelation on Original Series") 
    ax2 = fig.add_subplot(312)
    fig = plot_acf(timeseries.diff().dropna(), ax=ax2,  title="1st Order Differencing")
    ax3 = fig.add_subplot(313)
    fig = plot_acf(timeseries.diff().diff().dropna(), ax=ax3,  title="2nd Order Differencing")
    autoarima_model=None

    if len(extra_vars)>0:
        autoarima_model = pmd.auto_arima(timeseries[0:fit_range_max], X=extra_vars[0:fit_range_max], start_p=1,start_q=1,test="adf",trace=True, m=seasonal_m)
    else:
        #autoarima_model = pmd.auto_arima(timeseries[0:fit_range_max], X=None, start_p=1,start_q=1,test="adf",trace=True) #,m=seasonal_m) #kpss
        autoarima_model = pmd.auto_arima(timeseries[0:fit_range_max],information_criterion='aic', X=None, start_p=1,start_q=1, max_p=10, max_q=10,test="adf",trace=True, m=seasonal_m) #,m=seasonal_m)
        print
    fitted,confint = autoarima_model.predict(n_periods,return_conf_int=True,start=timeseries.index[-3])
    fittedv = autoarima_model.predict_in_sample(start=3)
    if verbose: print(fittedv)
    for i in range(0,3): fittedv=np.insert(fittedv,0,fittedv[0])

    index_of_fc=None
    if model_var=='close':
        index_of_fc = np.arange(timeseries.index[fit_range_max],timeseries.index[fit_range_max]+n_periods)
    else:
        index_of_fc = pd.date_range(timeseries.index[-1], periods = n_periods, freq=timescale)        
    # make series for plotting purpose
    if verbose: plt.show()
    fittedv_series = pd.Series(fittedv, index=timeseries[0:fit_range_max].index)
    fitted_series = pd.Series(fitted, index=index_of_fc)
    if verbose: print(fittedv_series - timeseries)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    if verbose:
        print(lower_series)
        print(fitted_series)
        print(upper_series)

    if verbose or True:
        # Plot
        plt.clf()
        plt.plot(timeseries)
        plt.plot(fitted_series, color='darkgreen')
        plt.plot(fittedv_series, color='yellow')
        plt.fill_between(lower_series.index, 
                lower_series, 
                upper_series, 
                color='k', alpha=.15)

        plt.title("SARIMA - Final Forecast of Stock prices - Time Series Dataset")
        plt.show()

# handle trades on the exit by updating the csv instructions
def HandleTradeExit(ticker, sale_price, buy_price, sale_date):
    out_dir_name='Instructions/out_*_instructions.csv'
    files_names_to_check = glob.glob(out_dir_name)
    for fname in files_names_to_check:
        try:
            dfnow = pd.read_csv(fname, sep=' ')
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            print(f'Could not load input csv: {fname}')
            dfnow=[]
        if len(dfnow)>0:
            #sell_date sold_at_loss sold
            dfnow.loc[dfnow['ticker']==ticker,'sell_date'] = sale_date
            dfnow.loc[dfnow['ticker']==ticker,'sold'] = sale_price
            dfnow.loc[dfnow['ticker']==ticker,'buy_limit'] = buy_price
            #dfnow[dfnow['sold']==ticker]['sold'] = sale_price
            if buy_price>sale_price:
                dfnow.loc[dfnow['ticker']==ticker,'sold_at_loss'] = buy_price
                #dfnow[dfnow['ticker']==ticker]['sold_at_loss'] = buy_price
            #dfnow['signal_date'] = pd.to_datetime(dfnow['signal_date'],errors='coerce')
# move old signal
def MoveOldSignals(api):
    out_dir_name='Instructions/out_*_instructions.csv'
    files_names_to_check = glob.glob(out_dir_name)
    for fname in files_names_to_check:
        try:
            dfnow = pd.read_csv(fname, sep=' ')
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            print(f'Could not load input csv: {fname}')
            dfnow=[]
        if len(dfnow)>0:
            out_df = []
            for t in dfnow['ticker'].values:

                positions = [p for p in api.list_positions() if p.symbol == t ]
                orders = [p for p in api.list_orders() if p.symbol == t ]
                print(t,positions,orders)
                time_of_signal = datetime.datetime.strptime(dfnow[dfnow['ticker']==t]['signal_date'].values[0],"%Y-%m-%dT%H:%M:%S-04:00")
                time_of_signal = time_of_signal.replace(tzinfo=est)
                # if more than 5 days, then let's remove it or replace it.
                if (time_of_signal<(datetime.datetime.now(tz=est)+datetime.timedelta(days=-5)) and len(positions)==0 and len(orders)==0 and dfnow[dfnow['ticker']==t]['sold_at_loss'].values[0]==0) or (time_of_signal<(datetime.datetime.now(tz=est)+datetime.timedelta(days=-40)) and len(positions)==0 and len(orders)==0 and dfnow[dfnow['ticker']==t]['sold_at_loss'].values[0]>0):
                    print(f'remove {t} from {fname} time of signal {time_of_signal}')
                    if len(out_df)==0:
                        out_df = pd.DataFrame(data=None, columns=dfnow.columns)
                    # add up those to remove
                    out_df = pd.concat([dfnow[dfnow['ticker']==t],out_df])
                    # remove from the dataframe
                    dfnow.drop(index=dfnow[dfnow['ticker']==t].index,inplace=True)

            # write out the results
            if len(out_df)>0:
                try:
                    fname_old = fname.replace('.csv','_old.csv')
                    if os.path.exists(fname_old):
                        dfold = pd.read_csv(fname_old, sep=' ')
                        out_df = pd.concat([dfold,out_df])
                        out_df.drop_duplicates(inplace=True)
                    out_df.to_csv(fname_old, sep=' ',index=False)
                    dfnow.to_csv(fname, sep=' ',index=False)
                except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
                    print(f'Could not load output csv OLD: {fname}')
            print(out_df)
            print(dfnow)
            #if dfnow['signal_date']
            #print(dfnow)
            #print(dfnow.dtypes)
            #print(dfnow.columns)
            #dfnow.to_csv(fname, sep=' ',index=False)
            
def DrawMinuteDiffs(minute_prices_thirty):

    #m1 =hour_prices_thirty[abs(hour_prices_thirty.minute_diff_now)>0.001]
    m1 =minute_prices_thirty[abs(minute_prices_thirty.minute_diff_now)>0.001]
    m1['minute_diff_filt'] = m1.minute_diff_now.shift(1)
    print(m1)
    print('prior was down')
    print(m1[m1['minute_diff_filt']>0].minute_diff_now.describe())
    print('prior was up')    
    print(m1[m1['minute_diff_filt']<0].minute_diff_now.describe())
    plt.scatter(m1.minute_diff_now,m1.minute_diff_filt.shift(1))
    plt.xlabel('Current change')
    plt.ylabel('Min ago change')
    plt.show()
    plt.scatter(m1.minute_diff_now,m1.minute_diff_5minfuture)
    plt.xlabel('Current change')
    plt.ylabel('5 Min future change')
    plt.show()
    plt.scatter(m1.minute_diff_now,m1.minute_diff_15minfuture)
    plt.xlabel('Current change')
    plt.ylabel('15 Min future change')
    plt.show()    
    
    plt.scatter(m1.minute_diff_now,m1.minute_diff_5minback)
    plt.xlabel('Current change')
    plt.ylabel('5 Min back change')
    plt.show()
    plt.scatter(m1.minute_diff_now,m1.minute_diff_15minback)
    plt.xlabel('Current change')
    plt.ylabel('15 Min back change')
    plt.show()    
    plt.scatter(minute_prices.minute_diff_now,minute_prices.minute_diff_2minago)
    plt.xlabel('Current change')
    plt.ylabel('Two Min ago change')
    plt.show()        
    plt.scatter(minute_prices.minute_diff_now,minute_prices.minute_diff_15minago)
    plt.xlabel('Current change')
    plt.ylabel('15 Min ago change')
    plt.show()

def FitWithBand(my_index, arr_prices, doMarker=True, ticker='X',outname='', poly_order = 2, price_key='adj_close',spy_comparison=[], doRelative=False, doJoin=True):
    """
    my_index : datetime array
    price : array of prices or values
    doMarker : bool : draw markers or if false draw line
    ticker : str : ticker symbol name
    outname : str : name of histogram to save as
    poly_order : int : order of polynomial to fit
    price_key : str : name of the price to entry to fit
    spy_comparison : array : array of prices to use as a reference. don't use when None
    doRelative : bool : compute the error bands with relative changes. Bigger when there is a big change in price
    doJoin : bool : join the two arrays on matching times
"""
    prices = arr_prices[price_key]
    x = mdates.date2num(my_index)
    xx = np.linspace(x.min(), x.max(), 1000)
    dd = mdates.num2date(xx)

    # prepare a spy comparison
    if len(spy_comparison)>0:
        if not doJoin:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.loc[arr_prices.index,:]
            prices /= (spy_comparison[price_key] / spy_comparison[price_key][-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'high'] /= (spy_comparison.high / spy_comparison.high[-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'low']  /= (spy_comparison.low  / spy_comparison.low[-1])
            arr_prices.loc[arr_prices.index==spy_comparison.index,'open'] /= (spy_comparison.open / spy_comparison.open[-1])
        else:
            arr_prices = arr_prices.copy(True)
            spy_comparison = spy_comparison.copy(True)
            arr_prices = arr_prices.join(spy_comparison,how='left',rsuffix='_spy')
            for i in ['high','low','open','close',price_key]:
                if len(arr_prices[i+'_spy'])>0:
                    arr_prices[i] /= (arr_prices[i+'_spy'] / arr_prices[i+'_spy'][-1])
            prices = arr_prices[price_key]

    # perform the fit
    z4 = np.polyfit(x, prices, poly_order)
    p4 = np.poly1d(z4)

    # create an error band
    diff = prices - p4(x)
    stddev = diff.std()
    print('p4:',z4[0],z4,x[-1],my_index[-1])
    print(-z4[1]/z4[0]/2.0)
    output_lines = '%s,%s,%s,%s' %(p4(x)[-1],stddev,diff[-1],prices[-1])
    if stddev!=0.0:
        output_lines = '%0.3f,%0.3f,%0.3f,%s,%s' %(p4(x)[-1],stddev,diff[-1]/stddev,prices[-1],p4)    
    if doRelative:
        diff /= p4(x)
        stddev = diff.std() #*p4(x).mean()
        
    pos_count_1sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-0.5*stddev))))
    pos_count_2sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-1.0*stddev))))
    pos_count_3sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-1.5*stddev))))
    pos_count_4sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-2.0*stddev))))
    pos_count_5sigma = len(list(filter(lambda x: (x >= 0), (abs(diff)-2.5*stddev))))
    if len(diff)>0:
        if debug: print('Time period: %s for ticker: %s' %(outname,ticker))
        coverage_txt='Percent covered\n'
        coverage = [100.0*pos_count_1sigma/len(diff) , 100.0*pos_count_2sigma/len(diff),
                        100.0*pos_count_3sigma/len(diff) , 100.0*pos_count_4sigma/len(diff),
                        100.0*pos_count_5sigma/len(diff) ]
        for i in range(0,5):
            if debug: print('Percent outside %i std. dev.: %0.2f' %(i+1,coverage[i]))
            coverage_txt+='%i$\sigma$: %0.1f\n' %(i+1,coverage[i])

    fig, cx = plt.subplots()
    if doRelative:
        stddev *= p4(x).mean()
        
    cx.errorbar(dd, p4(xx),
             np.ones(len(dd))*2.0*stddev,
             #color='k',
             ecolor='y',
             alpha=0.05,
             #label="4 sigma "
                    )
    cx.errorbar(dd, p4(xx),
             np.ones(len(dd))*1.5*stddev,
             #color='k',
             ecolor='y',
             alpha=0.1,
             #label="3 sigma "
                    )
    cx.errorbar(dd, p4(xx),
             np.ones(len(dd))*1.0*stddev,
             marker='.',
             color='k',
             ecolor='g',
             alpha=0.15,
             markerfacecolor='b',
             #label="2 sigma",
             capsize=0,
             linestyle='')
    cx.errorbar(dd, p4(xx),
             np.ones(len(dd))*0.5*stddev,
             marker='.',
             color='k',
             ecolor='g',
             alpha=0.2,
             markerfacecolor='b',
             #label="1 sigma",
             capsize=0,
             linestyle='')
    cx.plot(dd, p4(xx), '-g',label='Quadratic fit')

    plt.plot(arr_prices.high,color='red',label='High')
    plt.plot(arr_prices.low,color='cyan',label='Low')
    plt.plot(my_index,arr_prices.open, '+',color='orange',label='Open')

    if len(spy_comparison)>0:  cx.set_ylabel('Price / SPY')
    else: cx.set_ylabel('Price')
    #cx.set_xlabel('Date')
    
    if doMarker:
        cx.plot(my_index, prices, '+', color='b', label=price_key)
    else:
        plt.plot(prices, label=price_key)  
    plt.title("Fit - mean reversion %s for time period: %s" %(ticker,outname))
    plt.legend()
    plt.text(x.min(), max(prices), coverage_txt, ha='left', wrap=True)
    if draw: plt.show()
    if doPDFs: fig.savefig(outdir+'meanrev%s_%s.pdf' %(outname,ticker))
    fig.savefig(outdir+'meanrev%s_%s.png' %(outname,ticker))
    if not draw: plt.close()
    print('%s,%s,%s' %(ticker,outname,output_lines))
ticker='RDUS'
ticker='EYPT'
ticker='GGPI'
#ticker='VGR'
#ticker='X'
#ticker='TSLA'
ticker='SGMA'
ticker='PLBY'
ticker='WEN'
ticker='KR'
ticker='MSEX'
#ticker='SPY'
ticker='KZR'
#ticker='NVDA'
#ticker='SPY'
ticker='GGPI'
ticker='KTOS'
ticker='KZR'
#ticker='PLBY'
ticker='RENT'
ticker='IINN'
ticker='SPY'
filter_shift_days = 62
today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
todayFilter = (today + datetime.timedelta(days=-1*filter_shift_days))
d1 = todayFilter.strftime("%Y-%m-%dT%H:%M:%S-05:00")
thirty_days = (todayFilter + datetime.timedelta(days=-30)).strftime("%Y-%m-%dT%H:%M:%S-05:00")

# checking if it is shortable and tradeable:
aapl_asset = api.get_asset(ticker)
print(aapl_asset)
hour_prices_thirty    = runTicker(api, ticker, timeframe=TimeFrame.Hour, start=thirty_days, end=d1)
minute_prices_thirty  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
hour_prices_thirty_spy    = runTicker(api, 'SPY', timeframe=TimeFrame.Hour, start=thirty_days, end=d1)
minute_prices_thirty_spy  = runTicker(api, 'SPY', timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
hour_prices_10d       = GetTimeSlot(hour_prices_thirty,      days=10)
minute_prices_10d     = GetTimeSlot(minute_prices_thirty,    days=10)
hour_prices_spy_10d   = GetTimeSlot(hour_prices_thirty_spy,  days=10)
minute_prices_spy_10d = GetTimeSlot(minute_prices_thirty_spy,days=10)
#print(aapl_asset.tradable)
#print(aapl_asset.shortable)

ts = ALPHA_TIMESERIES()
sqlcursor = SQL_CURSOR()
daily_prices,j    = ConfigTable(ticker, sqlcursor,ts,'full',hoursdelay=18)
if filter_shift_days>0:
    daily_prices  = GetTimeSlot(daily_prices, days=6*365, startDate=todayFilter)
daily_prices_60d  = GetTimeSlot(daily_prices, days=60+filter_shift_days)
daily_prices_180d = GetTimeSlot(daily_prices, days=180+filter_shift_days)
daily_prices_365d = GetTimeSlot(daily_prices, days=365+filter_shift_days)
daily_prices_3y   = GetTimeSlot(daily_prices, days=3*365+filter_shift_days)
daily_prices_5y   = GetTimeSlot(daily_prices, days=5*365+filter_shift_days)
daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)
#print('ticker,time_span,fit_difference_in_stddev,current_price,stddev')
print('ticker,time_span,fit_expectations,stddev,fit_diff_significance,current_price')
sixtyday  = FitWithBand(daily_prices_60d.index, daily_prices_60d [['adj_close','high','low','open','close']],ticker=ticker,outname='60d')
print(sixtyday)

FitWithBand(daily_prices_180d.index,daily_prices_180d[['adj_close','high','low','open','close']],ticker=ticker,outname='180d')
FitWithBand(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],ticker=ticker,outname='365d')
FitWithBand(daily_prices_3y.index,  daily_prices_3y  [['adj_close','high','low','open','close']],ticker=ticker,outname='3y')
FitWithBand(daily_prices_5y.index,  daily_prices_5y  [['adj_close','high','low','open','close']],ticker=ticker,outname='5y', doRelative=False)

#print(daily_prices_365d[['adj_close','high','low','open','close']])
#print('before above and after below')
spy,j    = ConfigTable('SPY', sqlcursor,ts,'full',hoursdelay=18)
if filter_shift_days>0:
    spy  = GetTimeSlot(spy, days=6*365, startDate=todayFilter)
spy_daily_prices_60d  = GetTimeSlot(spy, days=60+filter_shift_days)
spy_daily_prices_365d = GetTimeSlot(spy, days=365+filter_shift_days)
spy_daily_prices_5y   = GetTimeSlot(spy, days=5*365+filter_shift_days)
FitWithBand(daily_prices_365d.index,daily_prices_365d[['adj_close','high','low','open','close']],
                ticker=ticker,outname='365dspycomparison',spy_comparison = spy_daily_prices_365d[['adj_close','high','low','open','close']])
FitWithBand(daily_prices_60d.index,daily_prices_60d[['adj_close','high','low','open','close']],
                ticker=ticker,outname='60dspycomparison',spy_comparison = spy_daily_prices_60d[['adj_close','high','low','open','close']])
FitWithBand(daily_prices_5y.index,daily_prices_5y[['adj_close','high','low','open','close']],
                ticker=ticker,outname='5yspycomparison',spy_comparison = spy_daily_prices_5y[['adj_close','high','low','open','close']])

# Spy comparison as well as mintue 10day comparison for minute and hour data
FitWithBand(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                ticker=ticker,outname='10dhspycomparison', price_key='close',spy_comparison = hour_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True)
FitWithBand(hour_prices_10d.index,hour_prices_10d[['high','low','open','close','vwap','volume']],
                ticker=ticker,outname='10dh', price_key='close')
FitWithBand(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                ticker=ticker,outname='10dmspycomparison', price_key='close',spy_comparison = minute_prices_spy_10d[['high','low','open','close','vwap','volume']],doJoin=True)
FitWithBand(minute_prices_10d.index,minute_prices_10d[['high','low','open','close','vwap','volume']],
                ticker=ticker,outname='10dm', price_key='close')

#print(daily_prices_365d[['adj_close','high','low','open','close']])
#print(spy_daily_prices_365d[['adj_close','high','low','open','close']])
#print(spy[['adj_close','high','low','open','close']])
#ARIMAauto(daily_prices_180d['adj_close'],extra_vars=[],model_var='adj_close',n_periods=10,fit_range_max=-1,seasonal_m=12)
#ACF(daily_prices_365d['adj_close'],50,sarima_order=(1, 1, 1,18),trend='t')
#ACF(daily_prices_365d['adj_close'],50,arima_order=(24, 1, 1),trend='t')
#ACF(daily_prices_180d['adj_close'],50,arima_order=(24, 1, 1),trend='t')
#ACF(daily_prices_365d['adj_close'],50)
#ARIMAauto(hour_prices_thirty['close'],extra_vars=[],model_var='close',n_periods=24,fit_range_max=len(hour_prices_thirty['close'])-1,seasonal_m=1) #250
#ARIMAauto(daily_prices_180d['daily_return'].dropna(),extra_vars=[],model_var='daily_return',n_periods=10,fit_range_max=-1,seasonal_m=7)

# see which stocks can be shorted? - DONE
# can you add the sector and total market? maybe a ratio to the SPY return? -> SPY is done. could consider sectors in the future
# could consider if this std dev should be a percentage change instead of dallar amount -> DONE. not huge


# also not a bad idea to have a minute and hourly setup for these or maybe momentum is better using -> DONE. seems better for short term trades. 
# These stocks are setup, but we need to process them...need a price,RMS,fit for 180d, 365, 3y and 5y. also hourly and minute-wise
#    - applying to finviz is great, but could also setup some kind of scan of prices to look for stocks exceeding the limits?
#    - Could have a list of stocks that are definitely worth buying when they drop (e.g. COST, KR, AAPL, GOOGL, AMZN, etc)

# how to trade?
#### on the 60d if there are excesses, then only buy on the side of more risk from the 1y.
#### use the 5y to guide the bigger trend. could use this to tune risk for how long to carry
#### could try to do a local fit or look for a volume spike to look for a max or min, but this could be a 2nd step. decide what time period matters?
# how do you know if the news matters?
# 
#JNYAX
