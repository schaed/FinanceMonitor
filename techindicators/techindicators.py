import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mplfinance import candlestick_ohlc
import mplfinance as mpf
import matplotlib.dates as mpl_dates
from numpy_ext import rolling_apply
#
# Simple Moving Average
# a is an array of prices, b is a period for averaging
def sma(a,b, sameSize=True):
    result = np.convolve(a, np.ones(b), 'valid') / b
    if sameSize:
        result = np.concatenate((np.zeros(b-1),result))
    return result
#
# Exponential Moving Average
# a is an array of prices, b is a period for averaging
def ema(a,b):
    ema_short = a.ewm(span=b, adjust=False).mean()
    ema_short = ema_short[b-1:]
    return ema_short
    #result = np.zeros(len(a)-b+1)
    #result[0] = np.sum(a[0:b])/b
    #for i in range(1,len(result)):
    #    result[i] = result[i-1]+(a[i+b-1]-result[i-1])*(2/(b+1))
    #print(ema_short-result)
    #return result
#
# Weighted Moving Average
# a is an array of prices, b is a period for averaging
def wma(a,b,sameSize=True):
    result = np.zeros(len(a)-b+1)
    for i in range(b-1,len(a)):
        result[i-b+1] = np.sum(np.arange(1,b+1,1)*a[i-b+1:i+1])\
        /np.sum(np.arange(1,b+1))
    if sameSize:
        result = np.concatenate((np.zeros(b-1),result))
    return result
#
# Kaufman's Adaptive Moving Average
# a is an array of prices, b is the period for the efficiency ratio
# c is the period for the fast EMA, d is the period for the slow EMA
def kama(a,b,c,d):
    fsc = 2/(c+1)# fast smoothing constant
    ssc = 2/(d+1)# slow smoothing constant
    er = np.zeros(len(a))# efficiency ratio
    pv = np.zeros(len(a))# periodic volatility
    pd = np.zeros(len(a))# price direction
    for i in range(1,len(a)):
        pv[i] = np.fabs(a[i]-a[i-1])
    for i in range(b,len(a)):
        pd[i] = np.fabs(a[i]-a[i-b])
    for i in range(b,len(a)):
        er[i] = pd[i]/np.sum(pv[i-b+1:i+1])
    sc = (er*(fsc-ssc)+ssc)**2
    result = np.zeros(len(a))
    result[b-1] = a[b-1]
    for i in range(b,len(a)):
        result[i] = result[i-1]+sc[i]*(a[i]-result[i-1])
    return result[b-1:]
#
# Average True Range
# a is array of high prices, b is array of low prices, 
# c is array of closing prices, d is period for averaging
def atr(a,b,c,d):

    df = pd.DataFrame(data={'tr0':a-b,'tr1':abs(a-c.shift()),'tr2':abs(b-c.shift())}, index=a.index)
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    matr = tr.ewm(alpha=1/d, adjust=False).mean()
    matr = matr[d-1:]
    return matr
    #tr = np.zeros(len(a))
    #result = np.zeros(len(a)-d+1)
    #tr[0] = a[0]-b[0]
    #for i in range(1,len(a)):
    #    hl = a[i]-b[i]
    #    hpc = np.fabs(a[i]-c[i-1])
    #    lpc = np.fabs(b[i]-c[i-1])
    #    tr[i] = np.amax(np.array([hl,hpc,lpc]))
    #result[0] = np.sum(tr[0:d])/d
    #for i in range(1,len(a)-d+1):
    #    result[i] = (result[i-1]*(d-1)+tr[i+d-1])/d
    #print(matr-result)
    #return result
#
# Relative Strength Index
# a is an array of prices, b is the period for averaging
def rsi(a,b,sameSize=True):
    change = np.zeros(len(a))
    gain = np.zeros(len(a))
    loss = np.zeros(len(a))
    ag = np.zeros(len(a))
    al = np.zeros(len(a))
    result = np.zeros(len(a))
    for i in range(1,len(a)):
        change[i] = a[i]-a[i-1]
        if change[i] == 0:
            gain[i] = 0
            loss[i] = 0
        if change[i] < 0:
            gain[i] = 0
            loss[i] = np.fabs(change[i])
        if change[i] > 0:
            gain[i] = change[i]
            loss[i] = 0
    ag[b] = np.sum(gain[1:b+1])/b# initial average gain
    al[b] = np.sum(loss[1:b+1])/b# initial average loss
    for i in range(b+1,len(a)):
        ag[i] = (ag[i-1]*(b-1)+gain[i])/b
        al[i] = (al[i-1]*(b-1)+loss[i])/b
    for i in range(b,len(a)):
        if al[i]!=0.0:
            result[i] = 100-100/(1+ag[i]/al[i])
        else:
            result[i] = 0.0

    if sameSize:
        for i in range(b):
            result[i] = None #result[b]
        return result
    return result[b:]
# 
# Commodity Channel Index
# a is array of high prices, b is array of low prices,
# c is array of closing prices, d is the number of periods
def cci(a,b,c,d,sameSize=True):

    TP = (a+b+c) / 3.0
    ma = TP.rolling(d).mean()
    mdev = (abs(TP - ma)).rolling(d).std(ddof=0)
    #print(abs(TP - ma))
    CCI = pd.Series((TP - ma) / (0.015 * mdev), name = 'CCI')
    return CCI
    #print(CCI)
    #data = data.join(CCI) 
    #return data
    # there is a difference in the typical price STDDEV. not sure which tpyical price to use
    #tp = (a+b+c)/3 # typical price
    #atp = np.zeros(len(a)) # average typical price
    #md = np.zeros(len(a)) # mean deviation
    #result = np.zeros(len(a))
    #for i in range(d-1,len(a)):
    #    atp[i] = np.sum(tp[i-(d-1):i+1])/d
    #    md[i] = np.sum(np.fabs(atp[i]-tp[i-(d-1):i+1]))/d
    #    result[i] = (tp[i]-atp[i])/(0.015*md[i])
    ##print(result)
    ##print(TP.rolling(d).mean() - atp)
    #print(mdev - md)
    #print(CCI - result)
    #if sameSize:
    #    return result
    #return result[d-1:]
#
# Accumulation/Distribution Line
# a is array of high prices, b is array of low prices,
# c is array of closing prices, d is the trading volume
def adl(a,b,c,d):
    mfm = ((c-b)-(a-c))/(a-b) # Money flow multiplier
    mfv = mfm*d # Money flow volume
    #result = np.zeros(len(a))
    #result[0] = mfv[0]
    #for i in range(1,len(a)):
    #    result[i] = np.sum(mfv[0:i+1])
    result = mfv.cumsum()
    #print(mfv_new - result)
    return result
#
# Moving Average Convergence/Divergence
# a is an array of prices, b is the numer of periods for fast EMA
# c is number of periods for slow EMA, 
# d is number of periods for signal line
def macd(a,b,c,d):
    line = ema(a,b)[c-b:]-ema(a,c)
    signal = ema(line,d)
    return line,signal
#
# Keltner Channels
# a, b, and c are high, low, and close price arrays
# d is numer of periods for center line EMA
# e is multiplier for ATR, and f is period for ATR
def kelt(a,b,c,d,e,f,sameSize=True):
    center = ema(c,d)
    if d>f:
        upper = center-e*atr(a,b,c,f)[d-f:]
        lower = center+e*atr(a,b,c,f)[d-f:]
    if f>=d:
        upper = center[f-d:]+e*atr(a,b,c,f)
        lower = center[f-d:]-e*atr(a,b,c,f)
    if sameSize:
        lower = np.concatenate((np.zeros(len(a)-len(lower)),lower))
        upper = np.concatenate((np.zeros(len(a)-len(upper)),upper))
        center = np.concatenate((np.zeros(len(a)-len(center)),center))
    return lower,center,upper

# takes in a data frame to make a rolling grouping
# df = market dataframes, w is the time period
def roll(df, w):
    # stack df.values w-times shifted once at each stack
    roll_array = np.dstack([df.values[i:i+w, :] for i in range(len(df.index) - w + 1)]).T
    # roll_array is now a 3-D array and can be read into
    # a pandas panel object
    panel = pd.Panel(roll_array, 
                     items=df.index[w-1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(w), name='roll'))
    # convert to dataframe and pivot + groupby
    # is now ready for any action normally performed
    # on a groupby object
    return panel.to_frame().unstack().T.groupby(level=0)
# compute beta
# df, market is the same array
# roll(df, 12).apply(beta)
def beta(df):
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1], df.columns[1:], name='Beta')

# compute rolling beta
# df, market is the first column if not specified, w=period
def rollingBeta(df, w, market=None, sameSize=True):
    if market!=None:
        min_length = min(len(market),len(df))        
        df_for_beta = pd.concat([market['adj_close'][-min_length:]] + [[df['adj_close'][-min_length:]]], axis=1).sort_index(1)
    betas = roll(df.pct_change().dropna(), w).apply(beta)
    difflen = len(df)-len(betas)
    if sameSize and difflen>0:
        betas = np.concatenate((np.zeros(difflen),betas))
    return betas

# compute rolling beta
# stock_data, market , w=period, assumes the column is adj_close
def rollingBetav2(stock_data, w, market_data, sameSize=True):
    min_length = min(len(market_data),len(stock_data)) 
    # Use .pct_change() only if joining Close data
    market_date_short = market_data['adj_close'][-min_length:]
    stock_data_short = stock_data['adj_close'][-min_length:]
    market_date_short = pd.DataFrame(market_date_short)
    stock_data_short = pd.DataFrame(stock_data_short)
    market_date_short.columns=['Market']
    stock_data_short.columns=['stock']
    stock = 'stock'
    beta_data = stock_data_short.join(market_date_short, how = 'inner').pct_change().dropna()
    ticker_covariance = beta_data.rolling(w).cov()
    # Limit results to the stock (i.e. column name for the stock) vs. 'Market' covariance
    ticker_covariance = ticker_covariance.loc[pd.IndexSlice[:, stock], 'Market'].dropna()
    benchmark_variance = beta_data['Market'].rolling(w).var().dropna()
    beta = ticker_covariance / benchmark_variance
    difflen = len(stock_data)-len(beta)
    if sameSize and difflen>0:
        beta = np.concatenate((np.zeros(difflen),beta))
    return beta

# compute rolling beta
# stock_data, market , w=period, assumes the column is adj_close
def rollingAlpha(stock_data, w, market_data, sameSize=True):
    min_length = min(len(market_data),len(stock_data)) 
    # Use .pct_change() only if joining Close data
    market_date_short = market_data['adj_close'][-min_length:]
    stock_data_short = stock_data['adj_close'][-min_length:]
    market_date_short = pd.DataFrame(market_date_short)
    stock_data_short = pd.DataFrame(stock_data_short)
    market_date_short.columns=['Market']
    stock_data_short.columns=['stock']
    stock = 'stock'
    beta_data = stock_data_short.join(market_date_short, how = 'inner').pct_change().dropna()
    ticker_covariance = beta_data.rolling(w).cov()
    # Limit results to the stock (i.e. column name for the stock) vs. 'Market' covariance
    ticker_covariance = ticker_covariance.loc[pd.IndexSlice[:, stock], 'Market'].dropna()
    benchmark_variance = beta_data['Market'].rolling(w).var().dropna()
    benchmark_mean = beta_data['Market'].rolling(w).mean().dropna()    
    stock_mean = beta_data[stock].rolling(w).mean().dropna()    
    beta = ticker_covariance / benchmark_variance
    alpha = stock_mean - beta*benchmark_mean    
    difflen = len(stock_data)-len(alpha)
    if sameSize and difflen>0:
        alpha = np.concatenate((np.zeros(difflen),alpha))
    return alpha

# calculate beta
# a is an array of daily percentage changes (e.g. pct_change), b is number of periods
def sharpe(a,b,sameSize=True):
    resultstd = rstd(a.dropna(),b, False)
    result = sma(a.dropna(),b,False)
    result /= resultstd
    difflen = len(a)-len(result)
    if difflen>0 and sameSize:
        result = np.concatenate((np.zeros(difflen),result))
    return result
#
# Rolling standard deviation
# a is an array of prices, b is number of periods
def rstd(a,b, sameSize=True):
    result = a.rolling(b).std(ddof=0)
    if not sameSize:
        result = result[b-1:]
    return result

# Polynomial Regression for two numpy arrays
def rsquare(daily_return,spy_return):
    covmat = np.cov(daily_return,spy_return)    
    beta = covmat[0,1]/covmat[1,1]
    alpha= np.mean(daily_return)-beta*np.mean(spy_return)

    ypred = alpha + beta * spy_return

    SS_res = np.sum(np.power(ypred-daily_return,2))
    SS_tot = covmat[0,0]*(len(daily_return)-1) # SS_tot is sample_variance*(n-1)
    r_squared = 1. - SS_res/SS_tot
    return r_squared

# compute rolling r-square, make sure you have daily_return and beta loaded
# df, market is the first column if not specified, w=period
def rollingRsquare(df, w, spy, sameSize=True):
    min_length = min(len(df),len(spy)) 
    # Use .pct_change() only if joining Close data
    dfs = df['daily_return'][-min_length:]
    spys = spy['daily_return'][-min_length:]
    dfs = pd.DataFrame(dfs)
    spys = pd.DataFrame(spys)
    rsq = rolling_apply(rsquare, w, dfs.daily_return.values, spys.daily_return.values)
    #rsq = df.rolling(w).apply(rsquare,raw=False)
    difflen = len(df)-len(rsq)
    if sameSize and difflen>0:
        rsq = np.concatenate((np.zeros(difflen),rsq))
    return rsq
#
# Bollinger Bands
# a is an array of prices, b is number of periods used
# for the SMA calculation, c is the multiplier for the 
# standard deviation, d is the number of periods for
# calculating the standard deviation
def boll(a,b,c,d,sameSize=True):
    stdboll = rstd(a,d,False)
    center = sma(a,b,False)
    if b>d:
        upper = center+c*stdboll[b-d:]
        lower = center-c*stdboll[b-d:]
    if d>=b:
        upper = center[d-b:]+c*stdboll
        lower = center[d-b:]-c*stdboll
    if sameSize:
        lower = np.concatenate((np.zeros(len(a)-len(lower)),lower))
        upper = np.concatenate((np.zeros(len(a)-len(upper)),upper))
        center = np.concatenate((np.zeros(len(a)-len(center)),center))
    return lower,center,upper
#
# Percentage price oscillator
# a is an array of prices, b is the numer of periods for fast EMA
# c is number of periods for slow EMA, 
# d is number of periods for signal line
def ppo(a,b,c,d):
    line = ((ema(a,b)[c-b:]-ema(a,c))/ema(a,c))*100
    signal = ema(line,d)
    return line,signal
#
# TRIX
# a is an array of prices, b is a period for each EMA in TRIX
# c is a period for the signal line EMA
def trix(a,b,c):
    triema = ema(ema(ema(a,b),b),b)
    line = np.zeros(len(triema)-1)
    for i in range(1,len(triema)):
        line[i-1] = ((triema[i]-triema[i-1])/triema[i-1])*100
    signal = ema(line,c)
    return line,signal
# 
# Stochastic oscillator
# a is an array of high prices, b is array of low prices,
# c is an array of closing prices, d is the look back period
# e is number of periods for %K SMA, f is the number of
# periods for %D SMA
def stoch(a,b,c,d,e,f,sameSize=True):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal 
    When the %K crosses below %D, sell signal
    """
    # Set minimum low and maximum high of the k stoch
    low_min  = b.rolling( window = d ).min()
    high_max = a.rolling( window = d ).max()
    # Fast Stochastic    
    k_fast = 100 * (c - low_min)/(high_max - low_min)
    pk = sma(k_fast,e)
    # Slow Stochastic
    pd = sma(pk,f)
    if not sameSize:
        pd = pd[d-1:]
        pk = pk[d-1:]
    return pk,pd
#
# Vortex indicator
# a is array of high prices, b is array of low prices
# c is array of close prices, d is number of periods
def vortex(a,b,c,d):
    tr = np.zeros(len(a))
    vp = np.zeros(len(a))
    vm = np.zeros(len(a))
    trd = np.zeros(len(a))
    vpd = np.zeros(len(a))
    vmd = np.zeros(len(a))
    tr[0] = a[0]-b[0]
    for i in range(1,len(a)):
        hl = a[i]-b[i]
        hpc = np.fabs(a[i]-c[i-1])
        lpc = np.fabs(b[i]-c[i-1])
        tr[i] = np.amax(np.array([hl,hpc,lpc]))
        vp[i] = np.fabs(a[i]-b[i-1])
        vm[i] = np.fabs(b[i]-a[i-1])
    for j in range(len(a)-d+1):
        trd[d-1+j] = np.sum(tr[j:j+d])
        vpd[d-1+j] = np.sum(vp[j:j+d])
        vmd[d-1+j] = np.sum(vm[j:j+d])
    trd = trd[d-1:]
    vpd = vpd[d-1:]
    vmd = vmd[d-1:]
    vpn = vpd/trd
    vmn = vmd/trd
    return vpn,vmn
#TODO add
# Average Directional Index (ADX)
# a is array of high prices, b is array of low prices
# c is array of close prices, d is number of periods
def adx(a,b,c,d, sameSize=True):
    tr = np.zeros(len(a))
    hph = np.zeros(len(a))
    pll = np.zeros(len(a))
    trd = np.zeros(len(a))
    pdm = np.zeros(len(a))
    ndm = np.zeros(len(a))
    pdmd = np.zeros(len(a))
    ndmd = np.zeros(len(a))
    for i in range(1,len(a)):
        hl = a[i]-b[i]
        hpc = np.fabs(a[i]-c[i-1])
        lpc = np.fabs(b[i]-c[i-1])
        tr[i] = np.amax(np.array([hl,hpc,lpc]))
        hph[i] = a[i]-a[i-1]
        pll[i] = b[i-1]-b[i]
    for j in range(1,len(a)):
        if hph[j]>pll[j]:
            if hph[j]>0:
                pdm[j]=hph[j]
        if pll[j]>hph[j]:
            if pll[j]>0:
                ndm[j]=pll[j]
    trd[d]=np.sum(tr[1:d+1])
    pdmd[d]=np.sum(pdm[1:d+1])
    ndmd[d]=np.sum(ndm[1:d+1])
    for k in range(d+1,len(a)):
        trd[k]=trd[k-1]-trd[k-1]/d+tr[k]
        pdmd[k]=pdmd[k-1]-pdmd[k-1]/d+pdm[k]
        ndmd[k]=ndmd[k-1]-ndmd[k-1]/d+ndm[k]
    trd = trd[d:]
    pdmd = pdmd[d:]
    ndmd = ndmd[d:]
    p = (pdmd/trd)*100
    n = (ndmd/trd)*100
    diff = np.fabs(p-n)
    summ = p+n
    dx = 100*(diff/summ)
    adx = np.zeros(len(dx))
    adx[d-1] = np.mean(dx[0:d])
    for l in range(d,len(dx)):
        adx[l] = (adx[l-1]*(d-1)+dx[l])/d
    adx = adx[d-1:]
    
    return p,n,adx

def getmaxposition(j):
    return len(j) - np.argmax(j) -1
def getminposition(j):
    return len(j) - np.argmin(j) -1
# 
# Aroon Oscillator
# a is array of high prices, b is array of low prices
# c is number of periods for calculation
def aroon(a,b,c, sameSize=True):
    up   = 100.0*(1.0 - (a.rolling(c).apply(getmaxposition))/c)
    down = 100.0*(1.0 - (a.rolling(c).apply(getminposition))/c)
    if not sameSize:
        up = up[c:]
        down = down[c:]
    return up,down,(up-down)
    #up = np.zeros(len(a))
    #down = np.zeros(len(a))
    #for i in range(c,len(a)):
    #    up[i] = 100*(1-(i-np.amax(np.where(a[0:i+1]==np.amax(a[i-c:i+1]))))/c)
    #    down[i] = 100*(1-(i-np.amax(np.where(b[0:i+1]==np.amin(b[i-c:i+1]))))/c)
    #up = up[c:]
    #down = down[c:]
    #print(up - upA)
    #print(down - downA)
    #return up,down,(up-down)
    
#
# Chandelier Exits
# a is array of high prices, b is array of low prices, 
# c is array of closing prices, d is number of periods
# e is multiplier for ATR, f is 'short' or 'long'
def chand(a,b,c,d,e,f):
    ch_atr = atr(a,b,c,d)
    maxp = np.zeros(len(a)-d+1)
    if f == 'long':
        for i in range(d-1,len(a)):
            maxp[i-d+1] = np.amax(a[i-d+1:i+1])
        result = maxp-ch_atr*e
    elif f == 'short':
        for i in range(d-1,len(a)):
            maxp[i-d+1] = np.amin(b[i-d+1:i+1])
        result = maxp+ch_atr*e
    else:
        print('The last parameter must be \'short\' or \'long\'')
    return result
#
# Rate of change (ROC)
# a is an array of prices, b is a number of periods
def roc(a,b,sameSize=True):
    #mroc = (a - a.shift(b))/a.shift(b)*100
    #if sameSize:
    #    return mroc
    #mroc = mroc[b:]
    #return mroc
    result = np.zeros(len(a)-b)
    for i in range(b,len(a)):
        result[i-b] = ((a[i]-a[i-b])/a[i-b])*100
    if sameSize:
        result = np.concatenate((np.zeros(b),result))
    #print(mroc - result)
    return result
#
# Coppock Curve
# a is an array of prices, b is number of periods for long ROC
# c is number of periods for short ROC, d is number of periods for WMA
def copp(a,b,c,d, sameSize=True):
    result = wma((roc(a,b,False)+roc(a,c,False)[b-c:]),d,False)
    if sameSize:
        result = np.concatenate((np.zeros(b+c-2),result))
    return result
#
# Force Index
# a is closing price, b is volume
# c is number of periods
def force(a,b,c):
    #FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex')
    ndays=1
    FI = a.diff(ndays)*b
    return ema(FI,c)
#
# Chaikin Money Flow (CMF)
# a is high prices, b is low prices
# c is closing prices, d is volume
# e is number of periods
def cmf(a,b,c,d,e,sameSize=True):
    mfv = (((c-b)-(a-c))/(a-b))*d
    resulta = mfv.rolling(e).sum()/d.rolling(e).sum()
    if not sameSize:
        resulta = resulta[e-1:]
    return resulta
    #result = np.zeros(len(a)-e+1)
    #for i in range(len(a)-e+1):
    #    result[i] = np.sum(mfv[i:i+e])/np.sum(d[i:i+e])
    #if sameSize:
    #    result = np.concatenate((np.zeros(e-1),result))
    #print(result - resulta)
    #return result

# 
# https://www.investopedia.com/terms/v/vwap.asp
# vwap - volume weighted trade. really only used during the day
# a is the price
# b is the high
# c is the low
# d is the volume
# e is the period to average over
def vwap(a,b,c,d,e):
    PV = d*(a+b+c)/3.0
    return PV.rolling(e).sum()/d.rolling(e).sum();

# 
# Chaikin Oscillator
# a is high prices, b is low prices
# c is closing prices, d is volume
# e is number of periods for short EMA
# f is number of periods for long EMA
def chosc(a,b,c,d,e,f):
    ch_adl = adl(a,b,c,d)
    ch_short = ema(pd.DataFrame(ch_adl),e)
    ch_long = ema(pd.DataFrame(ch_adl),f)
    return (ch_short[(f-e):]-ch_long)
#
# Ease of Movement (EMV)
# a is high prices, b is low prices
# c is volume, d is number of periods
def emv(a,b,c,d):
    #dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    #br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    #EVM = dm / br 
    #EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    dm = np.zeros(len(a)-1)
    for i in range(1,len(a)):
        dm[i-1] = (a[i]+b[i])/2 - (a[i-1]+b[i-1])/2
    br = ((c/100000000)/(a-b))
    br = br[1:]
    return sma(dm/br,d)
#
# Mass Index
# a is high prices, b is low prices
# c is a number of periods
def mindx(a,b,c):
    sema9 = ema((a-b),9)
    dema9 = ema(sema9,9)
    eratio = sema9[8:]/dema9
    result = np.zeros(len(eratio)-c+1)
    for i in range(len(eratio)-c+1):
        result[i] = np.sum(eratio[i:i+c])
    return result
#
# Money Flow Index (MFI)
# a is high prices, b is low prices
# c is closing prices, d is volume
# e is number of periods
def mfi(a,b,c,d,e):
    tp = (a+b+c)/3
    rmf = d*tp
    pmf = np.zeros(len(a))
    nmf = np.zeros(len(a))
    pmfs = np.zeros(len(a))
    nmfs = np.zeros(len(a))
    for i in range(1,len(a)):
        if tp[i]>tp[i-1]:
            pmf[i] = rmf[i]
        elif tp[i]<tp[i-1]:
            nmf[i] = rmf[i]
    for j in range(e,len(a)):
        pmfs[j] = np.sum(pmf[j-e+1:j+1])
        nmfs[j] = np.sum(nmf[j-e+1:j+1])
    pmfs = pmfs[e:]
    nmfs = nmfs[e:]
    return (100-100/(1+pmfs/nmfs))
#
# Negative Volume Index (NVI)
# a closing prices, b is volume
# c is number of periods
def nvi(a,b,c):
    ppc = np.zeros(len(a))
    pvc = np.zeros(len(a))
    line = np.zeros(len(a))
    line[0]=1000
    for i in range(1,len(a)):
        ppc[i]=100*((a[i]-a[i-1])/a[i-1])
        pvc[i]=100*((b[i]-b[i-1])/b[i-1])
        if pvc[i]<0:
            line[i]=line[i-1]+ppc[i]
        elif pvc[i]>0:
            line[i]=line[i-1]
    signal = ema(line,c)
    return line,signal
#
# On Balance Volume (OBV)
# a is closing prices, b is volume
def obv(a,b):
    obv = (np.sign(a.diff()) * b).fillna(0).cumsum()
    return obv
    #result = np.zeros(len(a))
    #for i in range(1,len(a)):
    #    if a[i]>a[i-1]:
    #        result[i]=result[i-1]+b[i]
    #    elif a[i]<a[i-1]:
    #        result[i]=result[i-1]-b[i]
    #    else:
    #        result[i]=result[i-1]
    #print(obv-result)
    #return result
#
# Percentage volume oscillator
# a is an array of volume, b is the numer of periods for fast EMA
# c is number of periods for slow EMA, 
# d is number of periods for signal line
def pvo(a,b,c,d):
    line = ((ema(a,b)[c-b:]-ema(a,c))/ema(a,c))*100
    signal = ema(line,d)
    return line,signal
#
# Pring's Know Sure Thing (KST)
# a is an array of prices 
# b, c, d, and e are periods for four rates of change
# f, g, h, and i are periods for moving averages of ROC
# j is the number of periods for signal line SMA
# standard parameters are (close price,10,15,20,30,10,10,10,15,9)
def kst(a,b,c,d,e,f,g,h,i,j):
    aroc1 = sma(roc(a,b),f)
    aroc2 = sma(roc(a,c),g)
    aroc3 = sma(roc(a,d),h)
    aroc4 = sma(roc(a,e),i)
    line = aroc1[len(aroc1)-len(aroc4):]+2*aroc2[len(aroc2)-len(aroc4):]+\
    3*aroc3[len(aroc3)-len(aroc4):]+4*aroc4
    signal = sma(line,j)
    return line,signal

# find the support levels using arcs with option 'low'
def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support
# find the resistance levels using arcs with option 'High'
def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance
# sets a threshold of 3% for the support levels.
def isFarFromLevel(l,levels):
    s=0.03
    return np.sum([abs(l-x)/x < s  for x in levels]) == 0
def getLevels(df):
    levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            levels.append((i,df['Low'][i]))
        elif isResistance(df,i):
            levels.append((i,df['High'][i]))
    return levels
def getMinLevels(df):
  levels = []
  for i in range(2,df.shape[0]-2):
    if isSupport(df,i):
      l = df['Low'][i]
      if isFarFromLevel(l,levels):
        levels.append((i,l))
    elif isResistance(df,i):
      l = df['High'][i]
      if isFarFromLevel(l,levels):
        levels.append((i,l))
  return levels

# get support levels as drawn h-lines
def supportLevels(data):
    df = data.loc[:, ['open', 'high', 'low', 'close','volume']]
    df.columns = ['Open', 'High', 'Low', 'Close','Volume']
    levels = getMinLevels(df)
    
    for level in levels:
        plt.hlines(level[1],xmin=df.index[level[0]], xmax=max(df.index),colors='blue')
        plt.text(max(df.index), level[1], ' %0.2f' %level[1], ha='left', va='center')
    return levels

# plot the support levels
def plot_support_levels(ticker,df,plots=[],outdir='',doPDF=True):
  
    levels = getMinLevels(df)
    #sline = []
    #for level in levels:
    #    sline+=[level[1]]
    #mpf.make_addplot(line80,panel='lower',color='r',secondary_y=False),
    fig,axes=mpf.plot(df, type='candle', style='charles',
            title=ticker,
            #hlines=dict(hlines=sline,colors=['b','b'],linestyle='-'),
            ylabel='Price ($) for %s' %ticker,
            ylabel_lower='Shares \nTraded',
            volume=True,
            mav=(200),
            returnfig=True,
            addplot=plots)
            #savefig=outdir+'test-mplfiance_support_'+ticker+'.pdf')
    
    axes[0].legend(['SMA200'])
    for level in levels:
        mylim = axes[0].get_xlim()
        axes[0].axhline(level[1],xmin=mylim[0]+10, xmax=mylim[1]-10,color='blue',linewidth=0.5,linestyle='-')
        axes[0].text(mylim[0]+5, level[1], ' %0.2f' %level[1], fontsize=8)
 #   fig.show()
    # Save figure to file
    fig.savefig(outdir+'test-mplfiance_support_'+ticker+'.png')
    if doPDF: fig.savefig(outdir+'test-mplfiance_support_'+ticker+'.pdf')
  #ax.xaxis.set_major_formatter(date_format)
  #fig.autofmt_xdate()
  #fig.tight_layout()
  #for level in levels:
  #  plt.hlines(level[1],xmin=df['Date'][level[0]], xmax=max(df['Date']),colors='blue')
  #fig.show()
