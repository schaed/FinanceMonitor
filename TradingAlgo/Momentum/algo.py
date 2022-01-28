#!/usr/bin/python
import requests
import time,os,sys
from ta.trend import macd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd

from ReadData import ALPACA_REST,ALPACA_STREAMCONN,runTicker
from alpaca_trade_api.rest import TimeFrame
from watchdog.observers import Observer
from file_change_monitor import MyFileHandler

# Load API
api = ALPACA_REST()
STOCK_DB_PATH = os.getenv('STOCK_DB_PATH')
global global_out_tickers
global_out_tickers = []
session = requests.session()

# We only consider stocks with per-share prices inside this range
min_share_price = 2.0
max_share_price = 70.0
# Minimum previous-day dollar volume for a stock we might consider
min_last_dv = 500000
# Stop limit to default to
default_stop = .95
# How much of our portfolio to allocate to any one position
risk = 0.001


class FileHandler(MyFileHandler):
    def  on_modified(self,event):

        # if boolean, then load them all. otherwise, see if the file was modified
        if type(event)==type(True):
            return

        for ifile_name in self.file_list:
            if ifile_name in event.src_path:
                return get_tickers()
            
class Stock:
    
    def __init__(self, ticker, asset, day_prices, prevday_volume, latest_trade):
        self.ticker = ticker
        self.asset = asset
        self.day_prices = day_prices
        self.prevday_close = day_prices['close'][-1]
        self.prevday_volume = prevday_volume
        self.latest_trade = latest_trade

    def Get1000m(self, api):
        """Get1000m: 
        api - alpaca rest api
        """
        #
        # Collect the latest 1000 minute bars
        #
        nyc = timezone('America/New_York')
        today = datetime.now(tz=nyc) 
        d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
        s1_day = (today + timedelta(days=-15)).strftime("%Y-%m-%dT%H:%M:%S-05:00")

        minute_prices = runTicker(api, self.ticker, timeframe=TimeFrame.Day, start=s1_day, end=d1, limit=100000)
        minute_prices = minute_prices[-1000:]
        
        return minute_prices
    
def get_1000m_history_data(symbols):
    print('Getting historical data...')
    minute_history = {}
    c = 0
    for symbol in symbols:
        minute_history[symbol.ticker] = symbol.Get1000m(api)
        c += 1
        print('{}/{}'.format(c, len(symbols)))
    print('Success.')
    return minute_history

def get_tickers():
    print('Getting current ticker data...')
    in_instruct_name=STOCK_DB_PATH+'/Instructions/out_momentum_instructions.csv'
    instruct = pd.read_csv(in_instruct_name,delimiter=' ')
    tickers = []
    if 'ticker' in instruct.columns:
        tickers = instruct['ticker'].unique()
    print(tickers)
    print('Success.')

    # Load assets
    assets = []
    for tick in tickers:
        assets +=[api.get_asset(tick)]

    symbols = [asset.symbol for asset in assets if asset.tradable]

    nyc = timezone('America/New_York')
    today = datetime.now(tz=nyc) 
    todayFilter = (today + timedelta(days=-2))
    d1 = today.strftime("%Y-%m-%dT%H:%M:%S-05:00")
    s1_day = (today + timedelta(days=-10)).strftime("%Y-%m-%dT%H:%M:%S-05:00")
    s1_min = (today).strftime("%Y-%m-%dT09:30:00-05:00")
    for s in symbols:
        trade = api.get_latest_trade(s)
        day_prices = runTicker(api, s, timeframe=TimeFrame.Day, start=s1_day, end=d1)
        #minute_prices = runTicker(api, s, timeframe=TimeFrame.Day, start=s1_min, end=d1)
        if len(day_prices)<2 or 'volume' not in day_prices.columns:
            continue
        prevday_volume = day_prices['volume'][-2]
        open_price = day_prices['open'][-1]
        if open_price<=0.0:
            continue

        if trade.p >= min_share_price and \
           trade.p <= max_share_price and \
           prevday_volume*trade.p > min_last_dv and \
           (open_price - trade.p)/open_price >= 0.035:
            my_stock = Stock(s, api.get_asset(tick), day_prices, prevday_volume, trade)
            found = False
            for ts in global_out_tickers:
                if s==ts.s:
                    found=True
            if not found:
                global_out_tickers+=[my_stock]
            
    #return global_out_tickers

def find_stop(current_value, minute_history, now):
    series = minute_history['low'][-100:] \
                .dropna().resample('5min').min()
    series = series[now.floor('1D'):]
    diff = np.diff(series.values)
    low_index = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0] + 1
    if len(low_index) > 0:
        return series[low_index[-1]] - 0.01
    return current_value * default_stop


def run(market_open_dt, market_close_dt, observer=None):
    # Establish streaming connection
    conn = ALPACA_STREAMCONN()

    # Update initial state with information from tickers
    volume_today = {}
    prev_closes = {}
    for ticker in global_out_tickers:
        symbol = ticker.ticker
        prev_closes[symbol] = ticker.prevday_close
        volume_today[symbol] = ticker.prevday_volume

    symbols = [ticker.ticker for ticker in global_out_tickers]
    print('Tracking {} symbols.'.format(len(symbols)))
    sys.stdout.flush()
    
    minute_history = get_1000m_history_data(global_out_tickers)

    portfolio_value = float(api.get_account().portfolio_value)

    open_orders = {}
    positions = {}

    # Cancel any existing open orders on watched symbols
    existing_orders = api.list_orders(limit=500)
    for order in existing_orders:
        if order.symbol in symbols:
            api.cancel_order(order.id)

    stop_prices = {}
    latest_cost_basis = {}

    # Track any positions bought during previous executions
    existing_positions = api.list_positions()
    for position in existing_positions:
        if position.symbol in symbols:
            positions[position.symbol] = float(position.qty)
            # Recalculate cost basis and stop price
            latest_cost_basis[position.symbol] = float(position.cost_basis)
            stop_prices[position.symbol] = (
                float(position.cost_basis) * default_stop
            )

    # Keep track of what we're buying/selling
    target_prices = {}
    partial_fills = {}

    # Use trade updates to keep track of our portfolio
    @conn.on(r'trade_update')
    async def handle_trade_update(conn, channel, data):
        symbol = data.order['symbol']
        last_order = open_orders.get(symbol)
        if last_order is not None:
            event = data.event
            if event == 'partial_fill':
                qty = int(data.order['filled_qty'])
                if data.order['side'] == 'sell':
                    qty = qty * -1
                positions[symbol] = (
                    positions.get(symbol, 0) - partial_fills.get(symbol, 0)
                )
                partial_fills[symbol] = qty
                positions[symbol] += qty
                open_orders[symbol] = data.order
            elif event == 'fill':
                qty = int(data.order['filled_qty'])
                if data.order['side'] == 'sell':
                    qty = qty * -1
                positions[symbol] = (
                    positions.get(symbol, 0) - partial_fills.get(symbol, 0)
                )
                partial_fills[symbol] = 0
                positions[symbol] += qty
                open_orders[symbol] = None
            elif event == 'canceled' or event == 'rejected':
                partial_fills[symbol] = 0
                open_orders[symbol] = None

    @conn.on(r'A$')
    async def handle_second_bar(conn, channel, data):
        symbol = data.symbol

        # First, aggregate 1s bars for up-to-date MACD calculations
        ts = data.start
        ts -= timedelta(seconds=ts.second, microseconds=ts.microsecond)
        try:
            current = minute_history[data.symbol].loc[ts]
        except KeyError:
            current = None
        new_data = []
        if current is None:
            new_data = [
                data.open,
                data.high,
                data.low,
                data.close,
                data.volume
            ]
        else:
            new_data = [
                current.open,
                data.high if data.high > current.high else current.high,
                data.low if data.low < current.low else current.low,
                data.close,
                current.volume + data.volume
            ]
        minute_history[symbol].loc[ts] = new_data

        # Next, check for existing orders for the stock
        existing_order = open_orders.get(symbol)
        if existing_order is not None:
            # Make sure the order's not too old
            submission_ts = existing_order.submitted_at.astimezone(
                timezone('America/New_York')
            )
            order_lifetime = ts - submission_ts
            if order_lifetime.seconds // 60 > 1:
                # Cancel it so we can try again for a fill
                api.cancel_order(existing_order.id)
            return

        # Now we check to see if it might be time to buy or sell
        since_market_open = ts - market_open_dt
        until_market_close = market_close_dt - ts
        if (
            since_market_open.seconds // 60 > 15 and
            since_market_open.seconds // 60 < 60
        ):
            # Check for buy signals

            # See if we've already bought in first
            position = positions.get(symbol, 0)
            if position > 0:
                return

            # See how high the price went during the first 15 minutes
            lbound = market_open_dt
            ubound = lbound + timedelta(minutes=15)
            high_15m = 0
            try:
                high_15m = minute_history[symbol][lbound:ubound]['high'].max()
            except Exception as e:
                # Because we're aggregating on the fly, sometimes the datetime
                # index can get messy until it's healed by the minute bars
                return

            # Get the change since yesterday's market close
            daily_pct_change = (
                (data.close - prev_closes[symbol]) / prev_closes[symbol]
            )
            if (
                daily_pct_change > .04 and
                data.close > high_15m and
                volume_today[symbol] > 30000
            ):
                # check for a positive, increasing MACD
                hist = macd(
                    minute_history[symbol]['close'].dropna(),
                    n_fast=12,
                    n_slow=26
                )
                if (
                    hist[-1] < 0 or
                    not (hist[-3] < hist[-2] < hist[-1])
                ):
                    return
                hist = macd(
                    minute_history[symbol]['close'].dropna(),
                    n_fast=40,
                    n_slow=60
                )
                if hist[-1] < 0 or np.diff(hist)[-1] < 0:
                    return

                # Stock has passed all checks; figure out how much to buy
                stop_price = find_stop(
                    data.close, minute_history[symbol], ts
                )
                stop_prices[symbol] = stop_price
                target_prices[symbol] = data.close + (
                    (data.close - stop_price) * 3
                )
                shares_to_buy = portfolio_value * risk // (
                    data.close - stop_price
                )
                if shares_to_buy == 0:
                    shares_to_buy = 1
                shares_to_buy -= positions.get(symbol, 0)
                if shares_to_buy <= 0:
                    return

                print('Submitting buy for {} shares of {} at {}'.format(
                    shares_to_buy, symbol, data.close
                ))
                try:
                    o = api.submit_order(
                        symbol=symbol, qty=str(shares_to_buy), side='buy',
                        type='limit', time_in_force='day',
                        limit_price=str(data.close)
                    )
                    open_orders[symbol] = o
                    latest_cost_basis[symbol] = data.close
                except Exception as e:
                    print(e)
                return
        if(
            since_market_open.seconds // 60 >= 24 and
            until_market_close.seconds // 60 > 15
        ):
            # Check for liquidation signals

            # We can't liquidate if there's no position
            position = positions.get(symbol, 0)
            if position == 0:
                return

            # Sell for a loss if it's fallen below our stop price
            # Sell for a loss if it's below our cost basis and MACD < 0
            # Sell for a profit if it's above our target price
            hist = macd(
                minute_history[symbol]['close'].dropna(),
                n_fast=13,
                n_slow=21
            )
            if (
                data.close <= stop_prices[symbol] or
                (data.close >= target_prices[symbol] and hist[-1] <= 0) or
                (data.close <= latest_cost_basis[symbol] and hist[-1] <= 0)
            ):
                print('Submitting sell for {} shares of {} at {}'.format(
                    position, symbol, data.close
                ))
                try:
                    o = api.submit_order(
                        symbol=symbol, qty=str(position), side='sell',
                        type='limit', time_in_force='day',
                        limit_price=str(data.close)
                    )
                    open_orders[symbol] = o
                    latest_cost_basis[symbol] = data.close
                except Exception as e:
                    print(e)
            return
        elif (
            until_market_close.seconds // 60 <= 15
        ):
            # Liquidate remaining positions on watched symbols at market
            try:
                position = api.get_position(symbol)
            except Exception as e:
                # Exception here indicates that we have no position
                return
            print('Trading over, liquidating remaining position in {}'.format(
                symbol)
            )
            api.submit_order(
                symbol=symbol, qty=position.qty, side='sell',
                type='market', time_in_force='day'
            )
            symbols.remove(symbol)
            if len(symbols) <= 0:
                conn.close()
            conn.deregister([
                'A.{}'.format(symbol),
                'AM.{}'.format(symbol)
            ])

    # Replace aggregated 1s bars with incoming 1m bars
    @conn.on(r'AM$')
    async def handle_minute_bar(conn, channel, data):
        ts = data.start
        ts -= timedelta(microseconds=ts.microsecond)
        minute_history[data.symbol].loc[ts] = [
            data.open,
            data.high,
            data.low,
            data.close,
            data.volume
        ]
        volume_today[data.symbol] += data.volume

    channels = ['trade_updates']
    for symbol in symbols:
        symbol_channels = ['A.{}'.format(symbol), 'AM.{}'.format(symbol)]
        channels += symbol_channels
    print('Watching {} symbols.'.format(len(symbols)))
    run_ws(conn, channels)


# Handle failed websocket connections by reconnecting
def run_ws(conn, channels):
    try:
        conn.run(channels)
    except Exception as e:
        print(e)
        conn.close()
        run_ws(conn, channels)

if __name__ == "__main__":
    # Get when the market opens or opened today
    nyc = timezone('America/New_York')
    today = datetime.today().astimezone(nyc)
    today_str = datetime.today().astimezone(nyc).strftime('%Y-%m-%d')
    calendar = api.get_calendar(start=today_str, end=today_str)[0]
    market_open = today.replace(
        hour=calendar.open.hour,
        minute=calendar.open.minute,
        second=0
    )
    market_open = market_open.astimezone(nyc)
    market_close = today.replace(
        hour=calendar.close.hour,
        minute=calendar.close.minute,
        second=0
    )
    market_close = market_close.astimezone(nyc)

    # reading in stocks
    event_handler = FileHandler(STOCK_DB_PATH+'Instructions/')
    observer = Observer(timeout=1)
    observer.schedule(event_handler,  STOCK_DB_PATH+'Instructions/',  recursive=True)
    observer.start()

    # Wait until just before we might want to trade
    current_dt = datetime.today().astimezone(nyc)
    since_market_open = current_dt - market_open
    while since_market_open.seconds // 60 <= 14:
        time.sleep(1)
        since_market_open = current_dt - market_open

    get_tickers()
    run(market_open, market_close)
    
    observer.stop()
    observer.join()
