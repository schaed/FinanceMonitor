#!/usr/bin/python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import alpaca_trade_api as alpaca
import asyncio
import pandas as pd
import numpy as np
import sys,glob,os,copy
import logging
import urllib3,requests

from ReadData import ALPACA_REST,ALPACA_STREAM,AddSMA,AddInfo,GetTimeSlot,FitWithBandMeanRev,AddData,slope,tValLinR,SQL_CURSOR,ALPHA_TIMESERIES,ConfigTable,runTicker
from alpaca_trade_api.rest import TimeFrame,APIError

import pytz,datetime
est = pytz.timezone('US/Eastern')
debug=False
logger = logging.getLogger()

def CheckTimeToExit(logger):
    # exit if it is too late into the evening
    if pd.Timestamp.now(tz='America/New_York').time() > pd.Timestamp('18:59',tz=est).time():
        logger.info(f'Closing program because the timestamp is late: %s' %pd.Timestamp.now(tz='America/New_York'))
        sys.exit(0)
        
def GetNewDF(df1,bar):
    # take a bar and a data frame and append one line for a bar
    df = pd.DataFrame([list(bar._raw.values())+[df1['i'][-1]+1]],columns=list(bar._raw.keys())+['i'])
    df.set_index('timestamp', inplace=True)
    if not df.empty:
        df.index = pd.to_datetime((df.index).astype('int64'), utc=True,).tz_convert(est)
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f'{df}')
    return pd.concat([df1,df])

# move old signals to new files.
def MoveOldSignals(api):
    out_dir_name='Instructions/out_*_instructions.csv'
    files_names_to_check = glob.glob(out_dir_name)
    for fname in files_names_to_check:
        try:
            dfnow = pd.read_csv(fname, sep=' ')
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            logger.error(f'Could not load input csv: {fname}')
            dfnow=[]
        if len(dfnow)>0:
            out_df = []
            for t in dfnow['ticker'].values:

                positions = [p for p in api.list_positions() if p.symbol == t ]
                orders = [p for p in api.list_orders() if p.symbol == t ]
                print(t,positions,orders)
                if dfnow[dfnow['ticker']==t]['signal_date'].values[0]=='signal_date':
                    logger.warning('removing this duplicate header line')
                    dfnow.drop(index=dfnow[dfnow['ticker']==t].index,inplace=True)
                    continue
                time_of_signal=''
                try:
                    time_of_signal = datetime.datetime.strptime(dfnow[dfnow['ticker']==t]['signal_date'].values[0],"%Y-%m-%dT%H:%M:%S-04:00")
                except:
                    print(dfnow[dfnow['ticker']==t]['signal_date'])
                    print('Error loading: %s' %(dfnow[dfnow['ticker']==t]['signal_date'].values[0]))
                    sys.stdout.flush()
                time_of_signal = time_of_signal.replace(tzinfo=est)
                # if more than 5 days, then let's remove it or replace it.
                if (time_of_signal<(datetime.datetime.now(tz=est)+datetime.timedelta(days=-5)) and len(positions)==0 and len(orders)==0 and dfnow[dfnow['ticker']==t]['sold_at_loss'].values[0]==0) or (time_of_signal<(datetime.datetime.now(tz=est)+datetime.timedelta(days=-40)) and len(positions)==0 and len(orders)==0 and dfnow[dfnow['ticker']==t]['sold_at_loss'].values[0]>0):
                    logger.info(f'remove {t} from {fname} time of signal {time_of_signal}')
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
                    logger.error(f'Could not load output csv OLD: {fname}')

def read_csv(in_file_name='/home/schae/testarea/FinanceMonitor/Instructions/out_meanrev_instructions.csv'):
    """read_csv - reads in the csv files and applies basic sanity checks like that the price is not already above the new recommendation
    Inputs:
    in_file_name - str - input csv file path like Instructions/out_bull_instructions_test.csv
    """
    fleet_out = {}
    
    if not os.path.exists(in_file_name):
        self._l.info(f'File path does not exist! {in_file_name}. Skipping...')
        return
    logger.info(f'next one: {in_file_name}')
        
    # reading in the sets of trades that we should be executing
    dfnow=[]

    try:
        dfnow = pd.read_csv(in_file_name, sep=' ')
        dfnow['sold_at_loss']  = dfnow.sold_at_loss.astype(bool)
        dfnow['timestamp']  = pd.to_datetime(dfnow.timestamp)
        # make sure the wash sale was more than 30 days ago
        dfnow = dfnow[dfnow.timestamp>(pd.Timestamp.now(tz=est)+datetime.timedelta(days=-32))]
    except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
        logger.warning(f'Could not load input csv other: {in_file_name}')
        dfnow=[]
    if len(dfnow)>0:
        for ticker in dfnow['symbol'].values:

            # check if this was sold at a loss. if so skip it
            if True in dfnow[dfnow['symbol']==ticker]['sold_at_loss'].values:
                logger.info(f'Skipping. this was already sold at a loss for ticker {ticker}')
                fleet_out[ticker] = True
            
    return fleet_out
                    
# handle trades on the exit by updating the csv instructions
def HandleTradeExit(ticker, sale_price, buy_price, sale_date, full_data, return_sold_at_loss_current=False):
    #out_dir_name='Instructions/out_*_instructions.csv'
    #files_names_to_check = glob.glob(out_dir_name)
    in_file_name='/home/schae/testarea/FinanceMonitor/Instructions/'
    files_names_to_check=[in_file_name+'out_meanrev_instructions.csv',]
    return_sold_at_loss = return_sold_at_loss_current
    for fname in files_names_to_check:
        try:
            dfnow = pd.read_csv(fname, sep=' ')
            sale_price = float(sale_price)
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            logger.error(f'Could not load input csv LS check: {fname} {sale_price}')
            dfnow=[]
        # fill the data to write out
        sold_at_loss = False
        if float(full_data.qty)>0 and buy_price>0: # this is buying to remove shares sold short
            sold_at_loss = (buy_price<float(sale_price))
        elif buy_price>0:
            sold_at_loss = (buy_price>float(sale_price))
        # if we already sold at a loss, then keep that designation
        if not return_sold_at_loss:
            return_sold_at_loss = sold_at_loss
        df_new = pd.DataFrame(data=[[ticker,
                                     ticker,
                                    full_data.event,
                                    full_data.order['side'],
                                    full_data.position_qty,
                                    full_data.price,
                                    full_data.qty,
                                    buy_price,
                                    sold_at_loss,
                                    full_data.order['updated_at'],
                                    full_data.execution_id]],
                              columns=['isymbol','symbol','fill_status','side','position_qty','sale_price','sale_qty','avg_entry_price','sold_at_loss','timestamp','execution_id'])
        if len(dfnow)==0:
            dfnow = df_new
        if len(dfnow)>0:
            #sell_date sold_at_loss sold
            dfnow = pd.concat([dfnow,df_new])
            dfnow.to_csv(fname, sep=' ',index=False)
    return return_sold_at_loss

class MeanRevAlgo:
    """ api is the contact to order stocks
        _ts: alpha vantage time series
        _sqlcursor: sql cursor
        _symbol : str the ticker symbol
        _lot : float the amount of cash to buy
        _limit : float the limit price
        _trail_percent : float the trailing percentage
        _take_profit : float gain percentage to sell. must be greater than 1
        _avg_entry_price : float average entry price
        _raise_stop : float gain percentage to raise the stop to ensure there is no loss
    """
    def __init__(self, api, ts, sqlcursor, spy, symbol, lot, limit, target, df=[]):
        self._api = api
        self._ts = ts
        self._sqlcursor = sqlcursor
        self._spy = spy
        self._symbol = symbol
        self._lot = lot
        self._limit = limit
        self._target = target
        self._trail_percent = 4.9
        self._stop_percent = 1.05
        self._take_profit=1.10
        self._raise_stop=1.01
        self._df = df
        self._bars = []
        self._state = ''
        self._avg_entry_price = -1.0
        self._fit_on_transaction = []
        self._l = logger.getChild(self._symbol)
        self.trade_side='buy'
        self._submit_init_order=False
        self.fig=[]
        self.fig_36d=[]
        self.sold_at_loss = False

        today = datetime.datetime.now(tz=est) 
        d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
        d1_set = today.strftime("%Y-%m-%d")
        #d1_set = "2022-01-19"
        #twelve_hours = (today + datetime.timedelta(hours=-12)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
        eighteen_days = (today + datetime.timedelta(days=-36)).strftime("%Y-%m-%dT%H:%M:%S-05:00")        
        minute_prices  = runTicker(self._api, self._symbol, timeframe=TimeFrame.Minute, start=eighteen_days, end=d1)
        minute_prices_thirty = minute_prices
        AddData(minute_prices_thirty)

        # try mean reversion
        minute_prices_thirty['adj_close']=minute_prices_thirty['close']
        minute_prices_thirty['sma200']=minute_prices_thirty['close']
        minute_prices_thirty['sma100']=minute_prices_thirty['close']
        minute_prices_thirty['sma50']=minute_prices_thirty['close']
        self.minute_prices_36d = minute_prices_thirty        
        self.minute_prices_18d = GetTimeSlot(self.minute_prices_36d,days=18,timez=est)
        self.minute_prices_60m = GetTimeSlot(self.minute_prices_36d,days=0,minutes=-60,timez=est)
        self._bars_since_fit=0
        self._fit()
        
        # collecting longer term checks for overbought or oversold
        daily_prices,j    = ConfigTable(self._symbol, self._sqlcursor,self._ts,'full',hoursdelay=22)
        try:
            start = time.time()
            daily_prices = AddInfo(daily_prices, self._spy, debug=debug)
            end = time.time()
            self._l.debug(f'Process time to add info: %s' %(end - start))
        except (ValueError,KeyError,NotImplementedError) as e:
            self._l.error("Testing multiple exceptions. {}".format(e.args[-1]))
            self._l.error('Error processing %s' %(self._symbol))
            #clean up
            #print('Removing: ',self._symbol)
            #self._sqlcursor.cursor().execute('DROP TABLE %s' %self._symbol)

        self.daily_prices_365d = GetTimeSlot(daily_prices,days=365)
        self.daily_prices_180d = GetTimeSlot(daily_prices,days=180)
        self.input_keysd = ['adj_close','high','low','open','close','sma200','sma100','sma50','sma20']
        self.fit_365d = FitWithBandMeanRev(self.daily_prices_365d.index,self.daily_prices_365d[self.input_keysd],ticker=self._symbol,outname='365d')
        self.fit_180d = FitWithBandMeanRev(self.daily_prices_180d.index,self.daily_prices_180d[self.input_keysd],ticker=self._symbol,outname='180d')
        self._l.info(f'180D: {self.fit_180d}')
        self._l.info(f'365D: {self.fit_365d}')
        p_now = self.minute_prices_18d['close'][-1]
        self.signif_180d = (p_now - self.fit_180d[0])/(self.fit_180d[1]/2)
        self.signif_365d = (p_now - self.fit_365d[0])/(self.fit_365d[1]/2)
        # Longer term trend indicates that this stock is oversold or overbought.
        #   - idea is not to fight the bigger trend to avoid a big mean reversion that goes opposite to the general trend
        self.no_long = (self.signif_180d)>3.0 or (self.signif_365d)>3.0;
        self.no_short =  (self.signif_180d)<-3.0 or (self.signif_365d)<-3.0;
        
        self._l.info(f'Significance dont go short: {self.no_short}')
        self._l.info(f'Significance dont go long: {self.no_long}')
        
        now = pd.Timestamp.now(tz='America/New_York').floor('1min')
        market_open = now.replace(hour=9, minute=30)
        today = now.strftime('%Y-%m-%d')
        tomorrow = (now + pd.Timedelta('1day')).strftime('%Y-%m-%d')
        self._update_status()
        self._init_state()

    def _fit(self):
        # fit the minute bars to extract the pol2 price prediction and error bands
        self.input_keys = ['adj_close','high','low','open','close','sma200','sma100','sma50']
        fig_tmp_36d = FitWithBandMeanRev(self.minute_prices_36d['i'], self.minute_prices_36d[self.input_keys], ticker=self._symbol,doDateKey=True, outname='36d',price_key='close')        
        fig_tmp = FitWithBandMeanRev(self.minute_prices_18d['i'], self.minute_prices_18d[self.input_keys], ticker=self._symbol,doDateKey=True, outname='18d',price_key='close')
        if type(fig_tmp)==type(None) or len(fig_tmp)==0 or np.isnan(fig_tmp[0]):
            self._l.error(f'Updating with nan...this is what we are fitting: {self.minute_prices_18d} {fig_tmp}. staying with other fit values')
        else:
            self.fig = fig_tmp
            self._l.info(f'Updating fit 18d: {self.fig}')
        if type(fig_tmp_36d)==type(None) or len(fig_tmp_36d)==0 or np.isnan(fig_tmp_36d[0]):
            self._l.error(f'Updating with nan...this is what we are fitting: {self.minute_prices_36d} {fig_tmp_36d}. staying with other fit values')
        else:
            self.fig_36d = fig_tmp_36d
            self._l.info(f'Updating fit 36d: {self.fig_36d}')            
        
        self._bars_since_fit=0
        
    def _init_state(self):

        symbol = self._symbol
        # Check that we have sufficient funds
        self._check_funds()
        # submit the order if requested
        if self._lot>0 and (self._order==None) and (self._position==None) and self._submit_init_order:
            self._l.warning(f'Init_state is submitting an order by request state {self._state} order {self._order}')            
            self._submit_buy()
        
        order = [o for o in self._api.list_orders() if o.symbol == symbol]
        position = [p for p in self._api.list_positions()
                    if p.symbol == symbol]

        self._order = order[0] if len(order) > 0 else None
        self._position = position[0] if len(position) > 0 else None
        if self._position is not None:
            if self._order is None:
                # TODO
                pass
                #self._state = 'TO_TRAILSTOP'
                #self._submit_trailing_stop()
            else:
                self._state = 'SELL_SUBMITTED'
                if self._order.side != 'sell':
                    self._l.warning(f'state {self._state} mismatch order {self._order}')
                if self._order.type == 'trailing_stop':
                    self._state = 'TRAILSTOP_SUBMITTED'
        else:
            if self._order is None and self._state!='':
                #self._state = 'TO_BUY'
                self._l.warning(f'Init_state is trying to submit an order state {self._state} order {self._order}')
                #self._submit_buy()
            elif self._order is not None:
                self._state = 'BUY_SUBMITTED'
                if self._order.side != 'buy':
                    self._l.warning(f'state {self._state} mismatch order {self._order}')
            elif self._state=='':
                    self._l.warning(f'No state {self._state} initialized {self._order} for {self._symbol}')
                    
    def _now(self):
        return pd.Timestamp.now(tz='America/New_York')

    def _update_status(self):
        self._update_orders()
        self._update_positions()
        if self._order!=None and self._position!=None :
            self._state = 'SELL_SUBMITTED'
        if self._order!=None and self._position==None :
            self._state = 'BUY_SUBMITTED'

    def _update_orders(self):
        self._order = [o for o in self._api.list_orders() if o.symbol == self._symbol]

    def _update_positions(self):
        self._position = [p for p in self._api.list_positions()
                    if p.symbol == self._symbol ]

    def _outofmarket(self):
        isOpen = True
        try:
            isOpen = self.api.get_clock().is_open
        except:
            pass
        return self._now().time() > pd.Timestamp('15:59',tz=est).time() and isOpen

    def _afterhours(self):
        return self._aftermarket() or self._premarket()
    
    def _aftermarket(self):
        return self._now().time() > pd.Timestamp('15:59',tz=est).time() and self._now().time() < pd.Timestamp('19:59',tz=est).time()
    
    def _premarket(self):
        return self._now().time() > pd.Timestamp('07:15',tz=est).time() and self._now().time() < pd.Timestamp('09:30',tz=est).time() 

    def _check_funds(self):
        # check how much money is available. Options: cash, buying_power, daytrading_buying_power
        my_account = self._api.get_account()
        if self._lot > float(my_account.cash):
            self._lot = float(my_account.cash)

    def checkup(self, position):
        # Check if anything has failed and we need to try submitting it again
        # self._l.info('periodic task')
        if self._state == 'FAIL_SELL' and self._order is None :
            self._submit_sell()
        if self._state == 'FAIL_TRAILSTOP' and self._order is None :
            self._submit_trailing_stop()
        if self._state == 'FAIL_BUY' and self._order is None:
            self._submit_buy()

    def _cancel_order(self):
        if self._order is not None:
            self._transition('TO_CANCEL')
            self._api.cancel_order(self._order.id)

    def on_bar(self, bar):
            #'open': bar.open,
            #'high': bar.high,
            #'low': bar.low,
            #'close': bar.close,
            #'volume': bar.volume,
        current_price = float(bar.close)

        # update with the latest bar
        self.minute_prices_36d = GetNewDF(self.minute_prices_36d,bar)
        self.minute_prices_18d = GetTimeSlot(self.minute_prices_36d,days=18,timez=est)
        self.minute_prices_60m = GetTimeSlot(self.minute_prices_36d,days=0,minutes=-60,timez=est)
        #self._l.info(f'{self.minute_prices_18d.i}')
        #self._l.info(f'{self.minute_prices_36d}')
        # could be smarter and only update the latest bar
        if True: # enable when used. currently not using this data
            AddData(self.minute_prices_60m)
        #self._l.info(f'{self.minute_prices_18d}')
        #print(self.minute_prices_18d.columns)

        # update the fit 
        self._bars_since_fit+=1
        if self._bars_since_fit>15:
            self._fit()
        # do not make decisions based on bad data
        if type(self.fig)==type(None) or len(self.fig)==0 or np.isnan(self.fig[0]):
            self._l.error(f'cannot make a decision without an 18d fit: {self.fig}')
            return
        if type(self.fig_36d)==type(None) or len(self.fig_36d)==0 or np.isnan(self.fig_36d[0]):
            self._l.error(f'cannot make a decision without an 36d fit: {self.fig_36d}') 
            return

        # have a position position, let's submit orders
        if self._position is not None:

            cost_basis = float(self._position.avg_entry_price)
            self._avg_entry_price = cost_basis

            # TODO figure out what to do with the limit price for the trailing stop
            limit_price = max([cost_basis * self._take_profit, current_price, self._target,self.fig[0]])
            self._l.info(f'Current price {current_price} and limit price {limit_price}, target: {self._target}')
            if float(self._position.qty)<0:
                limit_price = min([cost_basis / self._take_profit, current_price,self.fig[0]])
                if self._target>0:
                    limit_price = min([cost_basis / self._take_profit, current_price,self.fig[0],self._target])

            # evaluate how the fit could be used to set the sell price
            #if len(self._fit_on_transaction)>0:
            #    diff_mean_fit = self.fig[0]-self._fit_on_transaction[0] # the fit when it happens
            # not saving the transaction.
            if True:
                if self._order!=None and self._order.limit_price!=None:
                    self._limit = float(self._order.limit_price)
                # long position
                # side: buy or sell or short
                if float(self._position.qty)>0:
                    # fit mean is less than 0.1sigma from the entry, bailout. Exit as soon as possible
                    if self.fig[0] < (self._avg_entry_price + 0.1*self.fig[1] ) :
                        if current_price>= self._avg_entry_price or True:
                            if self._order==None or self._limit==None or (current_price>0.0 and abs(self._limit-current_price)/current_price>0.0033):
                                self._l.info(f'Sell low - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                                self._limit=current_price
                                self._cancel_order()
                                self._transition('TO_SELL')
                                self._submit_sell()
                        #else: # improve the exiting when we are losing
                        #    # if we have an order check that it is not already a market order or in the extended hours
                        #    if self._order!=None and (self._order.type!='market' or not self._order.extended_hours):
                        #        self._cancel_order()
                        #        self._l.info(f'Sell immediate - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                        #    if self._order==None:
                        #        self._transition('TO_SELL')
                        #        self._l.info(f'Sell immediate - bailout 0.1 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                        #        self._submit_sell(bailout=True)

                    # fit mean is less than 0.25sigma from the entry, then we exit with a profit if possible. Otherwise, just exit
                    elif self.fig[0] < (self._avg_entry_price + 0.25*self.fig[1] ) :

                        # update the limit price
                        new_limit_price=None                        
                        if current_price> self._avg_entry_price:
                            new_limit_price=current_price
                        else:
                            new_limit_price = self._avg_entry_price
                        
                        # check if we need to update the order
                        if self._order==None or self._limit==None or (self._limit>0.0 and abs(self._limit-new_limit_price)/self._limit>0.0033):
                            self._limit = new_limit_price
                            self._l.info(f'Sell 0.25 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                            self._cancel_order()
                            self._transition('TO_SELL')
                            self._submit_sell()
                    # fit mean is less than 0.75sigma from the entry; sell if we are making money
                    elif self.fig[0] < (self._avg_entry_price + 0.75*self.fig[1] ) and current_price> self._avg_entry_price:
                        new_limit_price = current_price
                        if self._order==None or self._limit==None or (current_price>0.0 and abs(self._limit-new_limit_price)/current_price>0.0033):
                            self._limit=new_limit_price
                            self._l.info(f'Sell 0.75 - Current price {current_price} and limit price {limit_price}, target: {self._target}') 
                            self._cancel_order()
                            self._transition('TO_SELL')
                            self._submit_sell()
                    else: # set limit order to the fit mean
                        # only submit if there is greater than .33 percent change
                        if (self._order==None) or self._order.limit_price==None or  (self._order!=None  and self._position!=None and abs(float(self._order.limit_price) - self.fig[0])/float(self._order.limit_price)>0.0033):
                            self._l.info(f'Sell limit order at fit mean update - Current price {current_price} and limit price {limit_price}, target: {self._target}')   
                            self._limit=self.fig[0]
                            self._cancel_order()
                            self._transition('TO_SELL')
                            self._submit_sell()
                        
                else: # short position
                    # fit mean is more than than 0.1sigma from the entry
                    if self.fig[0] > (self._avg_entry_price - 0.1*self.fig[1] ) :
                        new_limit_price=current_price                        
                        if current_price <= self._avg_entry_price:
                            if self._order==None or self._limit==None or (self._limit>0.0 and abs(self._limit-new_limit_price)/self._limit>0.0033):
                                self._limit=new_limit_price
                                self._l.info(f'Buy for short position - 0.1 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                                self._cancel_order()
                                self._transition('TO_SELL')
                                self._submit_sell()
                        #else: # TODO improve the exit procedure when we are losing
                        #    if self._order==None or self._limit==None or (self._limit>0.0 and abs(self._limit-new_limit_price)/self._limit>0.0033):
                        #    # if we have an order check that it is not already a market order or in the extended hours
                        #    if self._order!=None and (self._order.type!='market' or not self._order.extended_hours):
                        #        self._l.info(f'Cancelling order...in the process of bailing on the short position - 0.1 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                        #        self._cancel_order()
                        #    if self._order==None:
                        #        self._transition('TO_SELL')
                        #        self._l.info(f'Buy immediate for short position - bailout 0.1 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                        #        self._submit_sell(bailout=True)

                    # fit mean is more than 0.25sigma from the entry. Sell if we are making money; otherwise, put a limit order at the entry price
                    elif self.fig[0] > (self._avg_entry_price - 0.25*self.fig[1] ) :
                        new_limit_price = None
                        if current_price < self._avg_entry_price:
                            new_limit_price=current_price
                        else:
                            new_limit_price = self._avg_entry_price
                        
                        if self._order==None or self._limit==None or (self._limit>0.0 and abs(self._limit-new_limit_price)/self._limit>0.0033):
                            self._limit = new_limit_price
                            self._l.info(f'Buy for short position - 0.25 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                            self._cancel_order()
                            self._transition('TO_SELL')
                            self._submit_sell()
                    # fit mean is more than 0.75sigma from the entry. Sell if we are making making.
                    elif self.fig[0] > (self._avg_entry_price - 0.75*self.fig[1] ) and current_price < self._avg_entry_price:
                        
                        if self._order==None or self._limit==None or (current_price>0.0 and abs(self._limit-current_price)/current_price>0.0033):
                            self._limit=current_price
                            self._cancel_order()
                            self._l.info(f'Buy for short position - 0.75 - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                            self._transition('TO_SELL')
                            self._submit_sell()
                    else: # set limit order to the fit mean
                        # only submit if there is greater than .33 percent change
                        if (self._order==None) or self._order.limit_price==None or (self._order!=None and self._position!=None and abs(float(self._order.limit_price) - self.fig[0])/float(self._order.limit_price)>0.0033):
                            self._limit=self.fig[0]
                            self._l.info(f'Buy for short position - 0.75 - update order pricing - Current price {current_price} and limit price {limit_price}, target: {self._target}')
                            self._cancel_order()
                            self._transition('TO_SELL')
                            self._submit_sell()

            # TODO - setting up the trailing stops
            # if we clear 1%, then let's makes sure we don't lose.
            #if float(self._position.qty)>0 and self._avg_entry_price>0.0 and current_price>((self._avg_entry_price)*self._raise_stop) and self._trail_percent>1.0 and self._state == 'TRAILSTOP_SUBMITTED':
            #    self._trail_percent =1.0
            #    self._cancel_order()
            #    self._transition('TO_SELL')
            #    self._submit_trailing_stop()
            #if (float(self._position.qty)<0 and self._avg_entry_price>0.0 and current_price<((self._avg_entry_price)/self._raise_stop) and self._trail_percent>1.0 and self._state == 'TRAILSTOP_SUBMITTED'):
            #    self._trail_percent =1.0
            #    self._cancel_order()
            #    self._transition('TO_SELL')
            #    self._submit_trailing_stop()

            # exit positions if we cross the bail out threshold
            # - TODO improve the bailout procedure
            if float(self._position.qty)<0 and self._avg_entry_price>0.0 and current_price>((self._avg_entry_price)*self._stop_percent):
                self._l.info('Bailout! trying to get rid of our short')
                self._cancel_order()                
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)
            if float(self._position.qty)>0 and self._avg_entry_price>0.0 and current_price<((self._avg_entry_price)/self._stop_percent):
                self._l.info('Bailout! trying to sell')
                self._cancel_order()                
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)
                
            # selling for buy orders
            #if float(self._position.qty)>0 and current_price > limit_price:
            #    self._l.info(f'Submitting a sell order with current price {current_price} and limit price {limit_price}, cost_basis: {cost_basis}, target: {self._target}')
            #    if self._state == 'TRAILSTOP_SUBMITTED':
            #        self._cancel_order()
            #        self._transition('TO_SELL')
            #        self._submit_sell()
                    
            # if the price dips, then setup a trailing stop
            #if float(self._position.qty)>0 and current_price < cost_basis and self._state!='TRAILSTOP_SUBMITTED':
            #    self._cancel_order()
            #    self._transition('TO_SELL')
            #    self._submit_trailing_stop()
        # no position, so checking if we can add a position
        else:
            # run a quick check for nearby support lines
            #if len(self._df)>0 and trade.price<self._limit:
            #for ilimits in ['BolLower','sma20','downSL','vwap10']:
            #    if limit>0.0 and self._df[ilimits].values[0]<limit and abs(limit-self._df[ilimits].values[0])/limit<0.03:
            #        limit=self._df[ilimits].values[0]
            #print(self)

            # if an order already exists, then pass because the position hasn't collected yet
            #if self._order!=None and (self._position==None ):
            #    return
            # check the slope of the fit from the last two minutes
            slope_check = slope(self.fig[4],[self.fig[5],self.fig[5]+1])
            slope_check_36d = slope(self.fig_36d[4],[self.fig_36d[5],self.fig_36d[5]+1])
        
            # set these slope checks using historical data?
            # at 5d or 5*500min, then 1.5 sigma. add 0.5 for each day shorter than 0.5sigma
            switch_slope = 0.00006
            signif_hi=2.0
            signif_lo=1.5
            timeline=5*500
            timeline_36d=5*500
            # add the time to achieve a full reversion to the mean
            if slope_check_36d!=0:
                timeline_36d = (self.fig_36d[3]-self.fig_36d[0])/slope_check_36d
            if slope_check!=0:
                timeline = (self.fig[3]-self.fig[0])/slope_check
                if timeline>5*500 or timeline<0.0:
                    switch_slope = slope_check
                else:
                    signif_hi = 1.5+0.5*(5*500.0 - timeline)/500.0
                    if timeline_36d>0 and timeline_36d<5*500 and timeline>timeline_36d:
                        signif_hi = 1.5+0.5*(5*500.0 - timeline_36d)/500.0                        

                    # when there is downtrend in the slope, then set the significance at 1.5sigma
                    signif_lo = 1.5 #+0.5*(5*500.0 - timeline)/500.0

            # check that the 36d and the 18d agree
            signif_18d = self.fig[2]
            signif_36d = self.fig_36d[2]
            # if they point in the oppisite direction, then don't submit
            if signif_18d*signif_36d<0:
                self._l.info(f' - 18d and 36d have opposite signs 18d: {signif_18d} 36d: {signif_36d}')
                return
            if abs(signif_36d)<0.5:
                self._l.info(f' 36d indicates nothing is out of order - 18d: {signif_18d} 36d: {signif_36d}')
                return
                    
            # set the limit price                
            self.trade_side='buy'
            # if the fit slope is larger than the switch_slope, then check the significance. The significance is also less than 0 indicating below the mean price
            #   - the switch slope could be optimized -> just using the timeline to be more clear
            self._l.info(f'considering trade: signif 18d: %0.2f signif 36d: %0.2f signif_hi: %0.2f signif_lo: %0.2f timeline: %0.1f min 36d: %0.1f slope: %0.6f ' %(self.fig[2],self.fig_36d[2],signif_hi,signif_lo,timeline,timeline_36d,slope_check)+f'current_price: {current_price} fit: {self.fig}')
            if (timeline<5*500 and abs(self.fig[2])>signif_hi and self.fig[2]<0) or ((timeline>=5*500 or timeline<0) and abs(self.fig[2])>signif_lo and self.fig[2]<0):
                #trade = self._api.get_last_trade(self._symbol)
                max_price = current_price # max(current_price,trade.price)
                self._limit = int(100*max_price*1.002)/100.0
                self.trade_side='buy'
                if self.no_long:
                    self._l.info('long term (yearly) indicates this is already overbought')
                    return
                elif self._order!=None and (self._position==None ) and (self._limit>0 and abs(self._order.limit_price-self._limit)/self._limit>0.033):
                    self._l.info(f'Update Buy signal - Current price {current_price} and limit price {self._limit}, trade side: {self.trade_side} target: {self._target}')
                    self._cancel_order()
                    self._submit_buy()
                elif self._order==None and (self._position==None ):
                    self._l.info(f'Buy signal - Current price {current_price} and limit price {self._limit}, trade side: {self.trade_side} target: {self._target}') 
                    self._submit_buy()
                #print('over sold or bought!',self.fig,self.minute_prices_18d.index[-1],'minute slope: %0.3f' %self.minute_prices_18d['slope'][-1],' p4 slope: %0.4f' %slope(self.fig[4],[self.fig[5],self.fig[5]+1]))
            elif (self.fig[2]>signif_lo and (timeline>=5*500 or timeline<0)) or (self.fig[2]>signif_hi and timeline<=5*500) :
                #trade = self._api.get_last_trade(self._symbol)
                min_price = current_price #min(current_price,trade.price)                
                self._limit = int(100*min_price/1.002)/100.0
                self.trade_side='sell'
                if self.no_short:
                    self._l.info('long term (yearly) indicates this is already oversold')                    
                    return
                elif self._order!=None and (self._position==None ) and (self._limit>0 and abs(self._order.limit_price-self._limit)/self._limit>0.033):
                    self._l.info(f'Update Short signal - Current price {current_price} and limit price {self._limit}, trade side: {self.trade_side} target: {self._target}')
                    self._cancel_order()
                    self._submit_buy()
                elif self._order==None and (self._position==None ):
                    self._l.info(f'Short signal - Current price {current_price} and limit price {self._limit}, trade side: {self.trade_side} target: {self._target}')
                    self._submit_buy()                
                #print('over sold or bought!',self.fig,self.minute_prices_18d.index[-1],'minute slope: %0.3f' %self.minute_prices_18d['slope'][-1],' p4 slope: %0.4f' %slope(self.fig[4],[self.fig[5],self.fig[5]+1]))
                
        self._l.info( f'received bar start = {bar.timestamp}, close = {bar.close}, len(bars) = {len(self._bars)}')

        if self._outofmarket():
            return

    def on_order_update(self, event, order, full_data):
        self._l.info(f'order update: {event} = {order}')
        if event == 'fill':
            self._order = None
            #
            # Send signal to update the input file indicated what happened in the sale!
            self.sold_at_loss = HandleTradeExit(self._symbol, order['filled_avg_price'], self._avg_entry_price, order['filled_at'],full_data, return_sold_at_loss_current=self.sold_at_loss)
            if self._state == 'BUY_SUBMITTED':
                self._position = self._api.get_position(self._symbol)
                # TODO                
                #self._transition('TO_TRAILSTOP')
                #self._submit_trailing_stop()
                return
            elif self._state == 'TRAILSTOP_SUBMITTED':
                self._position = None
                self._transition('EXIT')
                self._l.info(f'exiting because position is sold with trailing stop order ')
                return
            elif self._state == 'SELL_SUBMITTED':
                self._position = None
                self._transition('EXIT')
                self._l.info(f'exiting because position is sold with limit order ')
                return
        elif event == 'partial_fill':
            self.sold_at_loss = HandleTradeExit(self._symbol, order['filled_avg_price'], self._avg_entry_price, order['filled_at'],full_data, return_sold_at_loss_current=self.sold_at_loss)
            self._position = self._api.get_position(self._symbol)
            self._order = self._api.get_order(order['id'])
            return
        elif event in ('canceled', 'rejected'):
            if event == 'rejected':
                self._l.warning(f'order rejected: current order = {self._order}')
                if self._state == 'TO_CANCEL':
                    self._cancel_order()
            if event == 'canceled':
                self._order = None
            if self._state == 'TO_CANCEL':
                self._l.info(f'expected cancel event for {event}: {self._state}')                
                pass
            elif self._state == 'BUY_SUBMITTED':
                if self._position is not None:
                    self._transition('TO_SELL')
                    self._submit_trailing_stop()
                else:
                    self._transition('TO_BUY')
            elif self._state == 'SELL_SUBMITTED':
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)
            #elif self._state == 'TRAILSTOP_SUBMITTED':
                #self._transition('TO_SELL')
                #self._submit_trailing_stop()
            else:
                self._l.warning(f'unexpected state for {event}: {self._state}')

    def _submit_buy(self):
        # avoid a wash sale
        if self.sold_at_loss:
            self._l.info(f'Cannot buy to avoid a wash sale!')
            return
        if self._position is not None:
            self._l.info(f'cannot submit buy with an existing position! {self._position}')
            return
            
        #print(self._api,self._symbol)
        trade = self._api.get_latest_trade(self._symbol)
        amount = int(self._lot / trade.price)
        limit = max(trade.price, self._limit)
        if self.trade_side=='sell':
            limit = min(trade.price, self._limit)
        limit = round(limit,2)
        # if the limit wasn't set, then lets exit
        if limit < 0:
            return

        extended_hours=False
        time_in_force = 'day'
        if self._afterhours():
            time_in_force = 'day' # opg - market on open or limit on open, fok : fill or kill, ioc: immediate or cancel (partial order)
            extended_hours=True
        self._transition('TO_BUY')
        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side=self.trade_side,
                type='limit',
                qty=amount,
                time_in_force=time_in_force,
                limit_price=limit,
                extended_hours=extended_hours,
                #take_profit=dict(limit_price=limit),
                #stop_loss=dict(
                #trail_percent=self._trail_percent
            )
            self._fit_on_transaction = copy.deepcopy(self.fig)
        except Exception as e:
            self._l.info(e)
            self._transition('FAIL_BUY')
            return

        self._order = order
        self._l.info(f'submitted buy {order}')
        self._transition('BUY_SUBMITTED')

    def _submit_trailing_stop(self):
        extended_hours=False
        time_in_force = 'day'
        if self._afterhours():
            time_in_force = 'day' # opg - market on open or limit on open, fok : fill or kill, ioc: immediate or cancel (partial order)
            extended_hours=True
            # trailing stops are not accepted afterhours
            return
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,
            type='trailing_stop',
            trail_percent=self._trail_percent,
            time_in_force=time_in_force,
            extended_hours=extended_hours,
        )

        try:
            order = self._api.submit_order(**params)
        except Exception as e:
            self._l.error(e)
            self._transition('FAIL_TRAILSTOP')
            return

        self._order = order
        self._l.info(f'submitted trailing stop {order}')
        self._transition('TRAILSTOP_SUBMITTED')
        
    def _submit_sell(self, bailout=False):
        if self._position is None:
            self._l.info(f'cannot submit sell if there is no position!')
            return
    
        # checking if this is extended hours
        extended_hours=False
        time_in_force = 'day'
        if self._afterhours():
            time_in_force = 'day' # opg - market on open or limit on open, fok : fill or kill, ioc: immediate or cancel (partial order)
            extended_hours=True
                
        # current price for the limit...get the last bar close
        current_price = float(self._api.get_latest_trade(self._symbol).price)
        p_now = current_price
        if len(self.minute_prices_18d)>0 and 'close' in self.minute_prices_18d:
            p_close_last_bar = self.minute_prices_18d['close'][-1]
        cost_basis = float(self._position.avg_entry_price)
        self._avg_entry_price = cost_basis
        #limit_price = round(max([cost_basis * self._take_profit, current_price, self._target,p_now]),2)
        limit_price = round(max([self._limit, current_price, self._target,p_close_last_bar]),2)
        self._l.info('%s' %([self._limit, current_price, self._target,p_close_last_bar]))
        trade_side='sell'
        qty_for_order = self._position.qty
        if float(self._position.qty)<0:
            trade_side='buy'
            qty_for_order = str(abs(int(qty_for_order)))
            limit_price = round(min([self._limit, current_price,p_close_last_bar]),2)
            if self._target>0:
                limit_price = round(min([self._limit, current_price,p_close_last_bar,self._target]),2)
            
        # creating sell params
        params = dict(
            symbol=self._symbol,
            side=trade_side,
            qty=qty_for_order,
            extended_hours=extended_hours,
            #replaces , # id number that this order replaces
            time_in_force=time_in_force)
        
        # final limit price update
        params.update(dict(type='limit',limit_price=limit_price))
        self._l.info(f'{params}')
        
        if bailout:
            if not self._afterhours():
                params['type'] = 'market'
                if 'limit_price' in params:
                    params['limit_price']=None

        # submiting orders
        try:
            order = self._api.submit_order(**params)
        except Exception as e:
            self._l.error(e)
            self._transition('FAIL_SELL')
            return

        self._order = order
        self._l.info(f'submitted sell {order}')
        self._transition('SELL_SUBMITTED')

    def _transition(self, new_state):
        self._l.info(f'transition from {self._state} to {new_state}')
        self._state = new_state


def main(args):
    stream = ALPACA_STREAM(data_feed='sip')
    api = ALPACA_REST()
    sys.stdout.flush()
    ts = ALPHA_TIMESERIES()
    STOCK_DB_PATH = os.getenv('STOCK_DB_PATH')
    sqlcursor = SQL_CURSOR('%s/stocksAV.db' %STOCK_DB_PATH)
    fleet = {}

    spy,j    = ConfigTable('SPY', sqlcursor,ts,'full',hoursdelay=22)
    spy = AddInfo(spy,spy,debug=debug)
    # check if this was already sold short
    sold_short = read_csv()
    
    symbols = args.symbols #.split(',')
    for symbol in symbols:
        # check for wash sales
        if symbol in sold_short and sold_short[symbol]:
            logger.info(f'This {symbol} has a wash sale, so we are skipping.')
            continue
        if args.lot>0:
            algo = MeanRevAlgo(api, ts, sqlcursor, spy, symbol, lot=args.lot, limit=args.limit, target=args.target, df=[])            
            fleet[symbol] = algo

    async def on_bars(data):
        if data.symbol in fleet:
            fleet[data.symbol].on_bar(data)

    for symbol in symbols:
        # check for wash sales
        if symbol in sold_short and sold_short[symbol]:
            logger.info(f'This {symbol} has a wash sale, so we are skipping.')
            continue
        
        logger.info(f'{symbol}')
        sys.stdout.flush()
        #stream.subscribe_trades(on_bars, symbol)
        stream.subscribe_bars(on_bars, symbol)
    
    async def on_trade_updates(data):
        logger.info(f'trade_updates {data}')
        symbol = data.order['symbol']
        if symbol in fleet:
            fleet[symbol].on_order_update(data.event, data.order, data)
    
    stream.subscribe_trade_updates(on_trade_updates)
    
    async def periodic():
        while True:
            CheckTimeToExit(logger)
            #if not api.get_clock().is_open:
            #    logger.info('exit as market is not open')
            #    sys.exit(0)
            await asyncio.sleep(30)
            positions = api.list_positions()
            for symbol, algo in fleet.items():
                pos = [p for p in positions if p.symbol == symbol]
                algo.checkup(pos[0] if len(pos) > 0 else None)
                
    loop = asyncio.get_event_loop()
    while 1:
        try:
            loop.run_until_complete(asyncio.gather(stream._run_forever(),periodic()))
        except (ConnectionResetError,urllib3.exceptions.ProtocolError,requests.exceptions.ConnectionError,APIError,ValueError,AttributeError,RuntimeError,TimeoutError):
            self._l.info(f'Connection error. will try to restart after 10s')
            time.sleep(10)
            pass
    loop.close()

if __name__ == '__main__':
    import argparse

    fmt = '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler('console.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    parser.add_argument('symbols', nargs='+')
    #parser.add_argument('--symbols', type=str, default='WEN',help='The amount of cash to spend')
    parser.add_argument('--lot', type=float, default=-1,help='The amount of cash to spend')
    parser.add_argument('--limit', type=float, default=-1,help='The limit price to buy')
    parser.add_argument('--target', type=float, default=-1,help='The target price to sell')

    main(parser.parse_args())
