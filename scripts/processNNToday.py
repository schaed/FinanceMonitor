from ReadData import SQL_CURSOR,GetUpcomingEarnings,AddInfo,ALPHA_TIMESERIES,ConfigTable,ALPHA_FundamentalData,GetTimeSlot,ApplySupportLevel,EarningsPreprocessing
import os,sys,time,datetime,copy,pickle
from keras.models import load_model
from sklearn import preprocessing
import numpy as np
debug =False

# results are shown in pred column. 2 is signal for a large price movement
def GetNNSelection(ticker, ts,connectionCal, sqlcursor, spy, debug=False,j=0,
                       training_dir='models/',
                       training_name='stockEarningsModelTestv2noEPS'):
    
    stock_info = EarningsPreprocessing(ticker, sqlcursor, ts, spy, connectionCal,j=j, ReDownload=False, debug=debug)

    COLS  = ['sma50r','sma20r','sma200r','copp','daily_return_stddev14',
    'beta','alpha','rsquare','sharpe','cci','cmf',
    'bop','SAR','adx','rsi10','ultosc','aroonUp','aroonDown',
    'stochK','stochD','willr',
    #'estimatedEPSr',
    'upSL','downSL','corr14',]
    
    model_filename = training_dir+'model'+training_name+'.hf'
    scaler_filename = training_dir+"scaler"+training_name+".save"
    scaler = pickle.load(open(scaler_filename, 'rb'))
    model = load_model(model_filename)
    
    X_test = stock_info[COLS] # use only COLS
    vector_y_pred = model.predict(X_test)
    stock_info['pred'] = np.argmax(vector_y_pred, axis = 1)
    if debug: print(stock_info[['adj_close','open','pred','sma50r','sma20r','sma200r','downSL','upSL']])
    return stock_info

ts = ALPHA_TIMESERIES()
connectionCal = SQL_CURSOR('earningsCalendar.db')
sqlcursor = SQL_CURSOR()
ticker='X'
j=0

# reading in the spy data
spy,j = ConfigTable('SPY', sqlcursor,ts,'full',hoursdelay=2)
AddInfo(spy,spy,debug=debug)

GetNNSelection(ticker, ts,connectionCal, sqlcursor, spy, debug=False,j=j,
                       training_dir='models/',
                       training_name='stockEarningsModelTestv2noEPS')
