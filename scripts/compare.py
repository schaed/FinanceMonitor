import alpaca_trade_api as tradeapi
import os
from ReadData import ALPACA_REST
#from alpaca_trade_api.rest import REST
#api = tradeapi.REST()
#export APCA_API_BASE_URL='https://paper-api.alpaca.markets'
#export APCA_API_BASE_URL='https://api.alpaca.markets'  # for live markets
#ALPACA_ID = os.getenv('ALPACA_ID')
#ALPACA_PAPER_KEY = os.getenv('ALPACA_PAPER_KEY')
#ALPHA_ID = os.getenv('ALPHA_ID')
api = ALPACA_REST()

#paper-api.alpaca.markets
ticker='X'
ticker='TSLA'

# Get our position in AAPL.
#aapl_position = api.get_position(ticker)

# Get a list of all of our positions.
portfolio = api.list_positions()
#print(portfolio)

# Print the quantity of shares for each position.
for position in portfolio:
    print("{} shares of {} market: {} cost_basis: {}".format(position.qty, position.symbol, position.market_value, position.cost_basis))


# Get a list of all of our history.
hist = api.get_portfolio_history(date_start='2021-07-01',date_end='2021-07-15') #,period='1D')
print(hist)

# Get a list of all of our orders that are open
orders = api.list_orders()
#print(orders)
for o in orders:
    print("{} shares of {} for {} on {} for {}".format(o.qty, o.symbol,o.filled_avg_price, o.filled_at, o.order_type))

print('')
print('Closed!')
# Get a list of all of our orders that are closed
orders = api.list_orders(status='closed',after='2021-07-01',limit=500)
print(len(orders))
for o in orders:
    print("{} shares of {} for {} on {} for {}".format(o.qty, o.symbol, o.filled_avg_price, o.filled_at, o.order_type))
