#!/usr/bin/python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import alpaca_trade_api as alpaca
import asyncio
import pandas as pd
import sys,glob,os
import logging
import urllib3,requests

from ReadData import ALPACA_REST,ALPACA_STREAM
from alpaca_trade_api.rest import TimeFrame,APIError

import pytz,datetime
est = pytz.timezone('US/Eastern')

logger = logging.getLogger()

# move old signals to new files.
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
                if dfnow[dfnow['ticker']==t]['signal_date'].values[0]=='signal_date':
                    print('removing this duplicate header line')
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

# handle trades on the exit by updating the csv instructions
def HandleTradeExit(ticker, sale_price, buy_price, sale_date):
    out_dir_name='Instructions/out_*_instructions.csv'
    files_names_to_check = glob.glob(out_dir_name)
    for fname in files_names_to_check:
        try:
            dfnow = pd.read_csv(fname, sep=' ')
            sale_price = float(sale_price)
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            print(f'Could not load input csv: {fname} {sale_price}')
            dfnow=[]
        if len(dfnow)>0:
            #sell_date sold_at_loss sold
            dfnow.loc[dfnow['ticker']==ticker,'sell_date'] = sale_date
            dfnow.loc[dfnow['ticker']==ticker,'sold'] = sale_price
            if buy_price>sale_price:
                dfnow.loc[dfnow['ticker']==ticker,'sold_at_loss'] = buy_price
            dfnow.to_csv(fname, sep=' ',index=False)
    
# Creates trades when requested
class  MyHandler(FileSystemEventHandler):
    def __init__(self, fleet,api,stream):
        FileSystemEventHandler.__init__(self)
        self.fleet = fleet # save a link to all of the trades
        self.api = api
        self.stream = stream
    def on_moved(self, event):
        print(f'event type: {event.event_type} path : {event.src_path}')
    def  on_modified(self,  event):
        #print(f'event type: {event.event_type} path : {event.src_path}')
        file_list=['Instructions/out_target_instructions.csv',
                       'Instructions/out_upgrade_instructions.csv',
                       #'Instructions/out_bull_instructions.csv',
                       'Instructions/out_pharmaphase_instructions.csv']
        # if boolean, then load them all. otherwise, see if the file was modified
        if type(event)==type(True):
            for ifile_name in file_list:
                self._read_csv(in_file_name=ifile_name)
        else:
            for ifile_name in file_list:
                if event.src_path==ifile_name:
                    self._read_csv(in_file_name=ifile_name)
                    
    def  on_created(self,  event):
        print(f'event type: {event.event_type} path : {event.src_path}')
    def  on_deleted(self,  event):
        print(f'event type: {event.event_type} path : {event.src_path}')

    def _read_csv(self, in_file_name='Instructions/out_target_instructions.csv'):
        """read_csv - reads in the csv files and applies basic sanity checks like that the price is not already above the new recommendation
        Inputs:
        in_file_name - str - input csv file path like Instructions/out_bull_instructions_test.csv
        """
        if not os.path.exists(in_file_name):
            print(f'File path does not exist! {in_file_name}. Skipping...')
            return
        
        # Defining bars to pass off to the class
        async def on_bars(data):
            if data.symbol in self.fleet:
                self.fleet[data.symbol].on_bar(data)
            
        # reading in the sets of trades that we should be executing
        dfnow=[]
        try:
            dfnow = pd.read_csv(in_file_name, sep=' ')
        except (ValueError,FileNotFoundError,ConnectionResetError,FileExistsError):
            print(f'Could not load input csv: {in_file_name}')
            dfnow=[]
        if len(dfnow)>0:
            for ticker in dfnow['ticker'].values:

                # check if this was sold at a loss. if so skip it
                if dfnow[dfnow['ticker']==ticker]['sold_at_loss'].values[0]!=0:
                    print(f'Skipping. this was already sold at a loss for ticker {ticker}')
                    continue
                # if it was already bought and sold, then skip.
                if dfnow[dfnow['ticker']==ticker]['sell_date'].values[0]!='X':
                    print(f'Skipping. this was already bought and sold for ticker {ticker}')
                    continue
                # Check that the buy limit is not near the upper bolanger
                my_limit = dfnow[dfnow['ticker']==ticker]['buy_limit'].values[0]
                my_UpperBol = dfnow[dfnow['ticker']==ticker]['BolUpper'].values[0]
                if my_limit > (my_UpperBol*0.98):
                    print(f'Skipping. this ticker {ticker} because the limit {my_limit} is near the upper bolanger {my_UpperBol}')
                    continue

                # if price target upgrade is close to the current price, then skip.
                if 'target_after' in  dfnow and in_file_name.count('target'):
                    target_after = dfnow[dfnow['ticker']==ticker]['target_after'].values[0]
                    if my_limit>0.0 and (target_after - my_limit)/my_limit>0.9:
                        print(f'Skipping. the price target {target_after} for this ticker is too small: {ticker}. previous close was {my_limit}')
                        continue
                
                if ticker not in self.fleet:
                    # add ticker to streaming
                    self.stream.subscribe_bars(on_bars, ticker)
                    
                    my_lot = dfnow[dfnow['ticker']==ticker]['lot'].values[0]
                    my_limit = dfnow[dfnow['ticker']==ticker]['buy_limit'].values[0]
                    my_target = max([dfnow[dfnow['ticker']==ticker]['upSL'].values[0],dfnow[dfnow['ticker']==ticker]['fivedaymax'].values[0],dfnow[dfnow['ticker']==ticker]['BolUpper'].values[0],my_limit*1.025])
                    self.fleet[ticker] = BullishAlgo(self.api, ticker, lot=my_lot, limit=my_limit, target=my_target, df = dfnow[dfnow['ticker']==ticker]);
class BullishAlgo:
    """ api is the contact to order stocks
        _symbol : str the ticker symbol
        _lot : float the amount of cash to buy
        _limit : float the limit price
        _trail_percent : float the trailing percentage
        _take_profit : float gain percentage to sell. must be greater than 1
        _avg_entry_price : float average entry price
        _raise_stop : float gain percentage to raise the stop to ensure there is no loss
    """
    def __init__(self, api, symbol, lot, limit, target, df=[]):
        self._api = api
        self._symbol = symbol
        self._lot = lot
        self._limit = limit
        self._target = target
        self._trail_percent = 4.0
        self._take_profit=1.03
        self._raise_stop=1.01
        self._df = df
        self._bars = []
        self._state = ''
        self._avg_entry_price = -1.0
        self._l = logger.getChild(self._symbol)

        now = pd.Timestamp.now(tz='America/New_York').floor('1min')
        market_open = now.replace(hour=9, minute=30)
        today = now.strftime('%Y-%m-%d')
        tomorrow = (now + pd.Timedelta('1day')).strftime('%Y-%m-%d')
        self._update_status()
        self._init_state()

    def _init_state(self):

        symbol = self._symbol
        # Check that we have sufficient funds
        self._check_funds()
        # submit the order if requested
        if self._lot>0 and (len(self._order)==0) and (len(self._position)==0):
            self._submit_buy()
        
        order = [o for o in self._api.list_orders() if o.symbol == symbol]
        position = [p for p in self._api.list_positions()
                    if p.symbol == symbol]
        self._order = order[0] if len(order) > 0 else None
        self._position = position[0] if len(position) > 0 else None
        if self._position is not None:
            if self._order is None:
                self._state = 'TO_TRAILSTOP'
                self._submit_trailing_stop()
            else:
                self._state = 'SELL_SUBMITTED'
                if self._order.side != 'sell':
                    self._l.warn(f'state {self._state} mismatch order {self._order}')
                if self._order.type == 'trailing_stop':
                    self._state = 'TRAILSTOP_SUBMITTED'
        else:
            if self._order is None:
                self._state = 'TO_BUY'
                self._submit_buy()
            else:
                self._state = 'BUY_SUBMITTED'
                if self._order.side != 'buy':
                    self._l.warn(f'state {self._state} mismatch order {self._order}')

    def _now(self):
        return pd.Timestamp.now(tz='America/New_York')

    def _update_status(self):
        self._update_orders()
        self._update_positions()
        if self._order!=None and len(self._order)>0 and self._position!=None and len(self._position)>0 :
            self._state = 'SELL_SUBMITTED'
        if self._order!=None and len(self._order)>0 and (self._position==None and len(self._position)==0) :
            self._state = 'BUY_SUBMITTED'
        
    def _update_orders(self):
        self._order = [o for o in self._api.list_orders() if o.symbol == self._symbol]
        #print(self._order)
        
    def _update_positions(self):
        self._position = [p for p in self._api.list_positions()
                    if p.symbol == self._symbol ]
    
    def _outofmarket(self):
        return self._now().time() >= pd.Timestamp('15:59').time()
    
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
            self._api.cancel_order(self._order.id)

    def on_bar(self, bar):
            #'open': bar.open,
            #'high': bar.high,
            #'low': bar.low,
            #'close': bar.close,
            #'volume': bar.volume,
        current_price = float(bar.close)
        
        # have a position position, let's submit orders
        if self._position is not None:

            cost_basis = float(self._position.avg_entry_price)
            self._avg_entry_price = cost_basis
            limit_price = max([cost_basis * self._take_profit, current_price, self._target])
            # if we clear 1%, then let's makes sure we don't lose.
            if self._avg_entry_price>0.0 and current_price>((self._avg_entry_price)*self._raise_stop) and self._trail_percent>1.0 and self._state == 'TRAILSTOP_SUBMITTED':
                self._trail_percent =1.0
                self._cancel_order()
                self._transition('TO_SELL')
                self._submit_trailing_stop()
            
            if current_price > limit_price:
                print(f'Submitting a sell order with current price {current_price} and limit price {limit_price}, cost_basis: {cost_basis}, target: {self._target}')
                if self._state == 'TRAILSTOP_SUBMITTED':
                    self._cancel_order()
                    self._transition('TO_SELL')
                    self._submit_sell()
            # if the price dips, then setup a trailing stop
            if current_price < cost_basis and self._state!='TRAILSTOP_SUBMITTED':
                self._cancel_order()
                self._transition('TO_SELL')
                self._submit_trailing_stop()
        self._l.info( f'received bar start = {bar.timestamp}, close = {bar.close}, len(bars) = {len(self._bars)}')

        if self._outofmarket():
            return

    def on_order_update(self, event, order):
        self._l.info(f'order update: {event} = {order}')
        if event == 'fill':
            self._order = None
            if self._state == 'BUY_SUBMITTED':
                self._position = self._api.get_position(self._symbol)
                self._transition('TO_TRAILSTOP')
                self._submit_trailing_stop()
                return
            elif self._state == 'TRAILSTOP_SUBMITTED':
                self._position = None
                self._transition('EXIT')                
                self._l.info(f'exiting because position is sold with trailing stop order ')
                #
                # Send signal to update the input file indicated what happened in the sale!
                HandleTradeExit(self._symbol, order['filled_avg_price'], self._avg_entry_price, order['filled_at'])
                return
            elif self._state == 'SELL_SUBMITTED':
                self._position = None
                self._transition('EXIT')
                # Send signal to update the input file indicated what happened in the sale!
                HandleTradeExit(self._symbol, order['filled_avg_price'], self._avg_entry_price, order['filled_at'])
                self._l.info(f'exiting because position is sold with limit order ')
                return
        elif event == 'partial_fill':
            self._position = self._api.get_position(self._symbol)
            self._order = self._api.get_order(order['id'])
            return
        elif event in ('canceled', 'rejected'):
            if event == 'rejected':
                self._l.warn(f'order rejected: current order = {self._order}')
            self._order = None
            if self._state == 'BUY_SUBMITTED':
                if self._position is not None:
                    self._transition('TO_SELL')
                    self._submit_trailing_stop()
                else:
                    self._transition('TO_BUY')
            elif self._state == 'SELL_SUBMITTED':
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)
            else:
                self._l.warn(f'unexpected state for {event}: {self._state}')

    def _submit_buy(self):
        trade = self._api.get_last_trade(self._symbol)
        amount = int(self._lot / trade.price)
        limit = min(trade.price, self._limit)

        # run a quick check for nearby support lines
        if len(self._df)>0 and trade.price<self._limit:
            for ilimits in ['BolLower','sma20','downSL','vwap10']:
                if limit>0.0 and self._df[ilimits].values[0]<limit and abs(limit-self._df[ilimits].values[0])/limit<0.03:
                    limit=self._df[ilimits].values[0]

        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='limit',
                qty=amount,
                time_in_force='day',
                limit_price=limit,
                #take_profit=dict(limit_price=limit),
                #stop_loss=dict(
                #trail_percent=self._trail_percent
            )
        except Exception as e:
            self._l.info(e)
            self._transition('FAIL_BUY')
            return

        self._order = order
        self._l.info(f'submitted buy {order}')
        self._transition('BUY_SUBMITTED')

    def _submit_trailing_stop(self):
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,
            type='trailing_stop',
            trail_percent=self._trail_percent,
            time_in_force='gtc',
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
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,
            time_in_force='gtc')
        if bailout:
            params['type'] = 'market'
        else:
            current_price = float(self._api.get_last_trade(self._symbol).price)
            cost_basis = float(self._position.avg_entry_price)
            self._avg_entry_price = cost_basis
            limit_price = max([cost_basis * self._take_profit, current_price, self._target])
            params.update(dict(type='limit',limit_price=limit_price))
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
    stream = ALPACA_STREAM()
    api = ALPACA_REST()
    fleet = {}

    # Move old signals so that we do not consider them
    MoveOldSignals(api)
    
    # checking for trades to execute!
    event_handler = MyHandler(fleet,api,stream)
    observer = Observer(timeout=1)
    observer.schedule(event_handler,  path='/Users/schae/testarea/finances/FinanceMonitor/Instructions/',  recursive=True)
    observer.start()

    symbols = args.symbols
    for symbol in symbols:
        if args.lot>0:
            algo = BullishAlgo(api, symbol, lot=args.lot, limit=args.limit, target=args.target, df=[])
            fleet[symbol] = algo

    # Trigger the loading of the trades
    event_handler.on_modified(True)

    async def on_bars(data):
        if data.symbol in fleet:
            fleet[data.symbol].on_bar(data)

    for symbol in symbols:
        #stream.subscribe_trades(on_bars, symbol)
        stream.subscribe_bars(on_bars, symbol)
    
    async def on_trade_updates(data):
        logger.info(f'trade_updates {data}')
        symbol = data.order['symbol']
        if symbol in fleet:
            fleet[symbol].on_order_update(data.event, data.order)
    
    stream.subscribe_trade_updates(on_trade_updates)
    
    async def periodic():
        while True:
            if not api.get_clock().is_open:
                logger.info('exit as market is not open')
                sys.exit(0)
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
            print('Connection error. will try to restart')
            pass
    loop.close()
    observer.stop()
    observer.join()

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
    parser.add_argument('--lot', type=float, default=-1,help='The amount of cash to spend')
    parser.add_argument('--limit', type=float, default=-1,help='The limit price to buy')
    parser.add_argument('--target', type=float, default=-1,help='The target price to sell')

    main(parser.parse_args())
