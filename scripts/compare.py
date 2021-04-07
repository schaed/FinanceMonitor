import alpaca_trade_api as tradeapi

#api = tradeapi.REST()
#export APCA_API_BASE_URL='https://paper-api.alpaca.markets'
#export APCA_API_BASE_URL='https://api.alpaca.markets'  # for live markets
ALPACA_ID = os.getenv('ALPACA_ID')
ALPACA_PAPER_KEY = os.getenv('ALPACA_PAPER_KEY')
ALPHA_ID = os.getenv('ALPHA_ID')
api = REST(ALPACA_ID,ALPACA_PAPER_KEY)

#paper-api.alpaca.markets
ticker='X'
ticker='TSLA'

# Get our position in AAPL.
aapl_position = api.get_position(ticker)

# Get a list of all of our positions.
portfolio = api.list_positions()

# Print the quantity of shares for each position.
for position in portfolio:
    print("{} shares of {}".format(position.qty, position.symbol))
