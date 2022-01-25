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
import AnaSignal

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
is_earnings=True
#s.Parse(inputTxt,'Honest Company', 'HON', sid=sid, nlp=nlp, is_earnings=False)
inputTxt='Bank of Hawaii reports Q4 EPS $1.55, consensus $1.35'
s.Parse(inputTxt,'Bank of Hawaii', 'BOH', sid=sid, nlp=nlp, is_earnings=False)
if len(s.PassEarnings())>0:
    AnaSignal.GenerateSignal(s.ticker,'out_earnings_instructions.csv',s.PassEarnings())
     
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
#HandleTradeExit('CUBI',0,0,'X')
#MoveOldSignals(api)
ticker='WRN'
#ticker='NTNX' #TDUP, SAVA, DOMO
#ticker='SAVA' #TDUP, SAVA, DOMO
#ticker='TSLA' #TDUP, SAVA, DOMO
#ticker='DOMO'
#ticker='KFY'
#ticker='X'
#ticker='HP'
#ticker='CVS'
#ticker='WBA'
#ticker='BSET'
ticker='X'
#ticker='CPE'
#ticker='GME'
#ticker='XELA'
#ticker='CLOV'
ticker='USEG'
ticker='DTSS'
ticker='WHLM'
ticker='YVR'
ticker='IKNA'
#ticker='NRBO'
#ticker='NURO'
#ticker='STAF'
ticker='LTRN'
ticker='CEMI'
ticker='FWP'
ticker='AMC'
ticker='BTBT'
ticker='INFI'
ticker='NTEC'
ticker='RDUS'
ticker='EYPT'
ticker='VGR'
ticker='GGPI'
ticker='KZR'
ticker='AVCT'
ticker='THRX'
ticker='GMTX'
ticker='MTCR'
ticker='VIR'
today = datetime.datetime.now(tz=est) #+ datetime.timedelta(minutes=5)
d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
thirty_days = (today + datetime.timedelta(days=-30)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
twohun_days = (today + datetime.timedelta(days=-200)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
hour_prices_thirty  = runTicker(api, ticker, timeframe=TimeFrame.Hour, start=twohun_days, end=d1)
AddData(hour_prices_thirty)
AddDataShort(hour_prices_thirty)
if True:
    minute_prices_thirty  = runTicker(api, ticker, timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
    minute_prices_spy  = runTicker(api, 'SPY', timeframe=TimeFrame.Minute, start=thirty_days, end=d1)
    # add the extra data
    AddData(minute_prices_thirty)
    AddData(minute_prices_spy)
    
    minute_prices_spy['signif_volume_spy'] = minute_prices_spy.signif_volume
    minute_prices_spy['change_spy'] = minute_prices_spy.change
    minute_prices_spy['minute_diffHL_spy'] = minute_prices_spy.minute_diffHL
    minute_prices_spy['minute_diff_spy'] = minute_prices_spy.minute_diff

    minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.signif_volume_spy,how='left')
    minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.change_spy,how='left')
    minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.minute_diff_spy,how='left')
    minute_prices_thirty = minute_prices_thirty.join(minute_prices_spy.minute_diffHL_spy,how='left')
    minute_prices_thirty['signif_volume_over_spy'] = minute_prices_thirty.signif_volume / minute_prices_thirty.signif_volume_spy
    minute_prices_thirty['change_over_spy'] = minute_prices_thirty.change - minute_prices_thirty.change_spy
    minute_prices_thirty['change_over_spy'] *=10.0
    minute_prices_thirty['change_over_spy'] -=1.0
    
    # get the last 5 days
    minute_prices = GetTimeSlot(minute_prices_thirty,days=7)
    minute_prices_spy_10d = GetTimeSlot(minute_prices_spy,days=10)
    #minute_prices=minute_prices_thirty
    print(minute_prices)
    print(minute_prices.describe())
    AddDataShort(minute_prices)
    AddDataShort(minute_prices_spy_10d)
    print(minute_prices.to_string())
    if False:
        DrawMinuteDiffs(minute_prices_thirty)
    
    plt.plot(minute_prices['volume'], label='volume')
    plt.plot(minute_prices['norm_open'],color='red', label='Perc Chg*10')
    #plt.plot(minute_prices['norm_close'],color='orange')
    #plt.plot(minute_prices['norm_high'],color='cyan')
    #plt.plot(minute_prices['norm_low'],color='cyan')
    plt.plot(minute_prices['ma50_div'],color='yellow', label='50m MA')
    plt.plot(minute_prices['minute_diff'],color='green', label='(C-O)/O')
    plt.plot(minute_prices['minute_diffHL'],color='cyan', label='(H-L)/O')
    plt.legend(loc="upper left")
    plt.show()
    
    
    print(minute_prices[minute_prices['signif_volume']>15.0])
    print('SPY')
    print(minute_prices_spy_10d[minute_prices_spy_10d['signif_volume']>15.0])
    
    minute_prices['signif_volume_over_spy']/=abs(minute_prices['signif_volume_over_spy']).max()*0.2
    minute_prices['signif_volume_div100'] = minute_prices['signif_volume']/20.0
    # check the ratio of the volume to that in spy
    plt.plot(minute_prices['signif_volume_div100'], label='vol signif',color='orange')
    #plt.plot(minute_prices['signif_volume_spy'], label='voume',color='cyan',alpha=0.25)
    plt.plot(minute_prices['signif_volume_over_spy'],label='sig Vol/SPY')
    plt.plot(minute_prices['minute_diff'],color='green', label='(C-O)/O')
    plt.plot(minute_prices['minute_diffHL'],color='cyan', label='(H-L)/O')
    plt.plot(minute_prices['norm_open'],color='red', label='Perc Chg*10')
    plt.plot(minute_prices['change_over_spy'],color='yellow', label='(return-spy)*10')
    plt.legend(loc="upper left")
    plt.show()

    # plot some correlations
    minute_prices['corr_signif_volume_spy'] = minute_prices['signif_volume'].rolling(60).corr(minute_prices.signif_volume_spy)
    minute_prices['corr_change_spy'] = minute_prices['change'].rolling(60).corr(minute_prices.change_spy)
    minute_prices['corr_minute_diff_spy'] = minute_prices['minute_diff'].rolling(60).corr(minute_prices.minute_diff_spy)
    minute_prices['corr_minute_diffHL_spy'] = minute_prices['minute_diffHL'].rolling(60).corr(minute_prices.minute_diffHL_spy)
    plt.plot(minute_prices['norm_open'],color='red', label='Perc Chg*10')
    plt.plot(minute_prices['corr_signif_volume_spy'],color='cyan', label='Corr Signif Vol')
    plt.plot(minute_prices['corr_minute_diff_spy']+1,color='green', label='Corr (C-O)/O')
    plt.plot(minute_prices['corr_minute_diffHL_spy']+2,color='yellow', label='Corr (H-L)/O')
    plt.plot(minute_prices['corr_change_spy']+3,color='orange', label='Corr Price Change')
    plt.legend(loc="upper left")
    plt.show()
    
    #print(minute_prices[1950:2000])
    #print(minute_prices[1950:2000].describe())
    #print(tValLinR(minute_prices.close[-10:-1],   extra_vars=minute_prices[-10:-1]))
    #print(tValLinR(minute_prices.close[1940:1950],extra_vars=minute_prices[1940:1950]))
    #print(tValLinR(minute_prices.close[1980:1990],extra_vars=minute_prices[1980:1990]))
    #print(tValLinR(minute_prices.close[1980:1990]))
    
    plt.plot(minute_prices.signif_volume)
    plt.show()
    #print(minute_prices[1850:1900])    
    #print(minute_prices[1950:2000])
    #print(minute_prices[1600:1650])
    #print(minute_prices[600:650])
    #print(minute_prices[1090:1140])
    
    #print(minute_prices[820:870])
    print(minute_prices[650:700])
    print(minute_prices[700:750])
    print(minute_prices[900:950])
    print(minute_prices[1000:1050])
    print(minute_prices[1050:1100])
    print(minute_prices[1250:1300])
        
    plt.plot(minute_prices['norm_open'],color='red')
    plt.plot(minute_prices['volume'])
    div = abs(minute_prices['slope'].max())/minute_prices['norm_open'].max()
    minute_prices['slope']/=div
    plt.plot(minute_prices['slope'],color='green')
    plt.show()
    #minute_prices_fit = minute_prices[['close','volume','open']] 
    #ARIMAauto(minute_prices['close'],extra_vars=minute_prices['volume'],model_var='close',n_periods=300,fit_range_max=1950)
    #ARIMAauto(minute_prices['close'],extra_vars=[],model_var='close',n_periods=300,fit_range_max=1890)
    #ARIMAauto(minute_prices['close'],extra_vars=[],model_var='close',n_periods=300,fit_range_max=839)
    #ARIMAauto(minute_prices['close'],extra_vars=[],model_var='close',n_periods=300,fit_range_max=810)
    ARIMAauto(minute_prices['close'],extra_vars=[],model_var='close',n_periods=300,fit_range_max=len(minute_prices['close'])-100,seasonal_m=1) # 300
    ARIMAauto(minute_prices['close'],extra_vars=[],model_var='close',n_periods=300,fit_range_max=len(minute_prices['close'])-1,  seasonal_m=1) # 300
    ACF(minute_prices['close'],50,timescale='none',arima_order=(6, 1, 1))
ts = ALPHA_TIMESERIES()
sqlcursor = SQL_CURSOR()
daily_prices,j = ConfigTable(ticker, sqlcursor,ts,'full')
daily_prices_180d = GetTimeSlot(daily_prices, days=180)
daily_prices_365d = GetTimeSlot(daily_prices, days=365)
#plt.plot(daily_prices_180d['adj_close'])
#plt.show()
daily_prices_180d['daily_return'] = daily_prices_180d['adj_close'].pct_change(periods=1)
ARIMAauto(daily_prices_180d['adj_close'],extra_vars=[],model_var='adj_close',n_periods=10,fit_range_max=-1,seasonal_m=12)
ACF(daily_prices_365d['adj_close'],50,sarima_order=(1, 1, 1,18),trend='t')
ACF(daily_prices_365d['adj_close'],50,arima_order=(24, 1, 1),trend='t')
ACF(daily_prices_180d['adj_close'],50,arima_order=(24, 1, 1),trend='t')
ACF(daily_prices_365d['adj_close'],50)
ARIMAauto(hour_prices_thirty['close'],extra_vars=[],model_var='close',n_periods=24,fit_range_max=len(hour_prices_thirty['close'])-1,seasonal_m=1) #250
ARIMAauto(daily_prices_180d['daily_return'].dropna(),extra_vars=[],model_var='daily_return',n_periods=10,fit_range_max=-1,seasonal_m=7)
