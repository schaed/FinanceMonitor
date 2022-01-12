import pandas as pd
import re,os,sys
from ReadData import SQL_CURSOR
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from Sentiment import Sentiment,News
import datetime,time
import wget,pickle
debug=False

# Add spacy word analyzer
import spacy
from spacy.tokens import Token
nlp = spacy.load('en_core_web_sm')

# create sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Import the alpaca setup to read in data
from ReadData import ALPACA_REST,ALPHA_TIMESERIES,is_date,runTickerAlpha,runTicker
from alpaca_trade_api.rest import TimeFrame
api = ALPACA_REST()

# Read the php or html. Then check for the short title as the key. build a struct.
def ParseTheFly(inputFileName='/tmp/recommend.php',my_map={},new_map={},is_earnings=False):
    pass
    #soup = BeautifulSoup(open(inputFileName,'r'), 'html.parser')
    

def Execute(f1='/tmp/recommend.php', f2='/tmp/news.php',f3='/tmp/earnings.php',total_news_map={},total_recs_map={}, outFileName='News.p'):

    new_news_map ={}
    new_recs_map ={}
    # read these in and save to a pickle file.
    ParseTheFly(inputFileName=f1,my_map=total_recs_map,new_map=new_recs_map)
    ParseTheFly(inputFileName=f3,my_map=total_recs_map,new_map=new_recs_map,is_earnings=True)
    ParseTheFly(inputFileName=f2,my_map=total_news_map,new_map=new_news_map)

    # Saving to a pickle file
    News = {'news':total_news_map,'recs':total_recs_map}
    pickle.dump( News, open( outFileName, "wb" ) )

    sys.stdout.flush()

def GetTicker(ex):
    j = ex.split(' ')
    if len(j)>1:
        return  pd.Series([j[0],' '.join(j[1:])])
    return pd.Series(['',''])
    
if __name__ == "__main__":
    # execute only if run as a script
    today = datetime.datetime.today()
    processList = [['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/commercial-services/','commerical_services'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/process-industries/','process_industries'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/communications/','communications'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/consumer-durables/','consumer_durables'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/health-technology/','health_technology'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/finance/','finance'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/non-energy-minerals/','non_energy_minerals'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/consumer-non-durables/','consumer_non_durables'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/distribution-services/','distribution_services'],
        ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/electronic-technology/','electronic_technology'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/energy-minerals/','energy_minerals'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/health-services/','health_services'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/industrial-services/','industrial_services'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/miscellaneous/','miscellaneous'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/producer-manufacturing/','producer_manufacturing'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/retail-trade/','retail_trade'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/technology-services/','technology_services'],
    ['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/transportation/','transportation'],
['https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/utilities/','utilities'],]

    # extra info
    # https://finviz.com/screener.ashx?v=122&o=industry&r=21
    # https://finviz.com/screener.ashx?v=162&o=industry&r=21
    # https://finviz.com/screener.ashx?v=132&o=industry&r=21
    #print(table_MN)
    #filename_rec = '/tmp/ccc.html'
    #table_MN = pd.read_html(filename_rec)
    #print(table_MN)

    psector=[]
    for p in processList:
        break;
        #URL = 'https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/commercial-services/'
        URL = p[0]
        filename_rec = '/tmp/%s.html' %p[1]
        os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))
        table_MN = pd.read_html(filename_rec)
        table_MN[0].columns = ['description','last','percent_change','change','rating','volume','market_cap','p_to_e','eps','employees','subsector']
        table_MN[0][['ticker','company']]=table_MN[0].description.apply(GetTicker)
        table_MN[0]['sector']=p[1]

        #.replace(['K','M'], [10**3, 10**6]).astype(int))
        table_MN[0]['percent_change'] = pd.to_numeric(table_MN[0].percent_change.str.rstrip('%'),errors='coerce') / 100.0
        print(table_MN[0].dtypes)
        table_MN[0]['volume'] = pd.to_numeric(table_MN[0].volume.replace({'K': 'e3', 'M': 'e6','B': 'e9','T': 'e12'}, regex=True),errors='coerce',downcast='integer')
        table_MN[0]['market_cap'] = pd.to_numeric(table_MN[0].market_cap.replace({'K': 'e3', 'M': 'e6','B': 'e9','T': 'e12'}, regex=True),errors='coerce',downcast='integer')
        table_MN[0]['employees'] = pd.to_numeric(table_MN[0].employees.replace({'K': 'e3', 'M': 'e6','B': 'e9','T': 'e12'}, regex=True),errors='coerce',downcast='integer')
        for j in ['p_to_e','eps']:
            table_MN[0][j] = pd.to_numeric(table_MN[0][j],errors='coerce')
        print(table_MN[0].dtypes)            
        print(table_MN[0][['last','percent_change','volume','employees','market_cap','p_to_e']])            
        if len(psector)==0:
            psector = table_MN[0]
        else:
            psector = pd.concat([psector,table_MN[0]])
    #sqlcursorSectorShort = SQL_CURSOR(db_name='partialSector.db')        
    #psector.to_sql('partialSectors',sqlcursorSectorShort,index=False,if_exists='append')

    
    #sys.exit(0)
    total_table=[]    
    for i in range(0,406):
    #for i in range(0,2):        
        print(i)
        URL = 'https://finviz.com/screener.ashx?v=111\&o=industry\&r=%s' %(1+i*20)
        filename_rec='/tmp/cc%i.html' %i
        if not os.path.exists(filename_rec):
            os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))
        
        table_MN = pd.read_html(filename_rec)
        for t in table_MN:
            #print(t.columns)
            if len(t)>0 and 4 in t.columns and t.loc[0,3]=='Sector':
                t.columns = t.loc[0].tolist()
                t=t.drop(0)
                t=t.drop('No.',axis=1)
                print(t.dtypes)
                print(t)
                t.Change = pd.to_numeric(t.Change.str.rstrip('%'),errors='coerce') / 100.0
                for a in ['Market Cap','P/E','Price','Volume']:
                    t[a] = pd.to_numeric(t[a].str.strip('%').replace({'K': 'e3', 'M': 'e6','B': 'e9','T': 'e12'}, regex=True),errors='coerce',downcast='integer')
                print(t)
                if len(total_table)==0:
                    total_table = t
                else:
                    total_table = pd.concat([total_table,t])
    print(total_table.dtypes)
    sqlcursorFull = SQL_CURSOR(db_name='sectorInfo.db')
    total_table.to_sql('sectors', sqlcursorFull, index=False,if_exists='append')
#https://finviz.com/screener.ashx?v=111&f=sec_basicmaterials&r=221
