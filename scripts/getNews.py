import pandas as pd
import re,os,sys
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

    soup = BeautifulSoup(open(inputFileName,'r'), 'html.parser')
    #print(soup.prettify())
    #print(soup.attrs)
    #print(soup.dt)
    #only_a_tags = SoupStrainer("a")
    #only_tags_with_id_link2 = SoupStrainer(id="link2")
    #attrs={"class":re.compile("fundPriceCell\d+")}
    #a = pd.read_html('recommend.php',attrs={"class":re.compile("fundPriceCell\d+"),'class':re.compile("ticker\d+")})
    #a = pd.read_html('recommend.php') #,attrs={"section":'class'})

    td = soup.find_all('td')

    tickers = []
    timeSlot = ''
    completeText=''
    shortText=''
    company=''
    debug=False
    upgradeRel = []
    downgradeRel = []
    nochangeRel = []
    for t in td:
        tickers = []
        timeSlot = ''
        completeText=''
        shortText=''
        currPrice=''
        company=''
        upgradeRel = []
        downgradeRel = []
        nochangeRel = []
        #print(t.prettify())
        #continue
        allD = t.find_all(attrs={'class':'completeText'})
        for d in allD:
            allP = d.find_all('p')
            if len(allP)>0:
                completeText=allP[0].get_text()
            if debug:
                for p in allP:
                    print(p.get_text())

        allU = t.find_all(attrs={'class':'upgrade relatedRec'})
        for u in allU:
            upgradeRel+=[u.get_text().strip()]
            if debug: print(u.get_text())
        allU = t.find_all(attrs={'class':'downgrade relatedRec'})
        for u in allU:
            downgradeRel+=[u.get_text().strip()]
            if debug: print(u.get_text())
        allU = t.find_all(attrs={'class':'no_change relatedRe'})
        for u in allU:
            nochangeRel+=[u.get_text().strip()]
            if debug: print(u.get_text())
        allS = t.find_all(attrs={'class':'statsCompany'})
        if len(allS)>0:
            currPrice = allS[0].get_text().strip()
            currPrice = currPrice.split(' ')
            if len(currPrice)>0:
                currPrice = currPrice[0].lstrip('\$')
                try:
                    currPrice = float(currPrice)
                except:
                    currPrice = currPrice
        if debug:
            for s in allS:
                print(s.get_text())
        #print(t.prettify())

        allTime = t.find_all('span',attrs={'class':'fpo_overlay soloHora'})
        if len(allTime)>0:
            timeSlot = allTime[0].get_text()
            if debug: print(timeSlot[-8:])
            timeSlot = datetime.datetime.strptime(timeSlot[-8:-2]+'20'+timeSlot[-2:]+' '+timeSlot[:-8], '%m/%d/%Y %H:%M')

        allA = t.find_all('a',attrs={'class':'newsTitleLink'})
        for a in allA:
            shortText=a.get_text()
            if debug:
                print(a.get_text())
            #print(a.prettify())
            allT = t.find_all('span',attrs={'class':'ticker fpo_overlay'})
            for t in allT:
                my_ticker = t.get('data-ticker')
                if my_ticker:
                    if debug: print(my_ticker)
                    tickers+=[my_ticker]
            allC = t.find_all('p',attrs={'class':'infoCompany'})
            if len(allC)>0:
                company = allC[0].get_text()
            if debug:
                for c in allC:
                    print(c.get_text())

        if company!='':
            my_news = News(tickers,timeSlot,
                 completeText.rstrip('Reference Link').strip().rstrip('\n'),
                 shortText,company,currPrice,upgradeRel,downgradeRel,nochangeRel,is_earnings=is_earnings)
            if debug: print(my_news)

            # only process new stories
            if shortText not in my_map:
                my_map[shortText] = my_news
                new_map[shortText] = my_news

                # running the sentiment analysis
                print('')
                print(shortText)
                print(my_news.Sentiment(sid=sid,nlp=nlp,is_earnings=is_earnings))
            if debug:
                print(my_news)
                print(tickers)
                print(timeSlot)
                print(completeText.rstrip('Reference Link').strip().rstrip('\n'))
                print('shortText: %s' %shortText)
                print(company)
                print(currPrice)        
                print(upgradeRel)
                print(downgradeRel)
                print(nochangeRel)

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
    
if __name__ == "__main__":
    # execute only if run as a script
    today = datetime.datetime.today()
    
    # all news stories from today and load yesterday if it exists
    total_news_map ={}
    total_recs_map ={}
    outFileName='News/News_%s_%s_%s.p' %(today.day,today.month,today.year)
    #outFileName='Newstoday.p'
    if os.path.exists(outFileName): # and False:
        try:
            #[total_news_map,total_recs_map] = pickle.load( open( outFileName, "rb" ) )
            oldNews = pickle.load( open( outFileName, "rb" ) )
            if 'news' in oldNews: total_news_map = oldNews['news']
            if 'recs' in oldNews: total_recs_map = oldNews['recs']
            if debug: print(total_recs_map.keys())
        except:
            print('Could not read the older news file: %s' %outFileName)
    while (today.hour<23 or (today.hour==23 and today.minute<30)):
        try:
        #if True:
            #after the fact https://thefly.com/news.php?market_mover_filter=on&h=5
            #maybe others https://www.americanbankingnews.com/category/market-news/analyst-articles-us/page/17
            #maybe finviz.com
            #https://thefly.com/news.php?earnings_filter=on&h=3
            URL = 'https://thefly.com/news.php?analyst_recommendations=on&h=2'
            filename_rec = '/tmp/recommend.php'
            os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))

            URL = 'https://thefly.com/news.php?earnings_filter=on&h=3'
            filename_earn = '/tmp/earnings.php'
            os.system('wget -T 30 -q -O %s %s' %(filename_earn,URL))
            
            URL = 'https://thefly.com/news.php'
            filename_news = '/tmp/newsInfo.php' # https://thefly.com/news.php?earnings_filter=on&h=3 earnings
            os.system('wget -T 30 -q -O %s %s' %(filename_news,URL))
            #filename_news='/tmp/breakingtest.html'
            # download the results
            Execute(f1=filename_rec, f2=filename_news, f3=filename_earn, total_news_map=total_news_map,total_recs_map=total_recs_map,outFileName=outFileName)
        except:
            print('Error downloading pages!')
        
        # sleep for 5 minutes
        time.sleep(300)
        today = datetime.datetime.today()

    
    #Execute(f1='test/recommend.php', f2='test/news.php',total_news_map=total_news_map,total_recs_map=total_recs_map)
    # probably need to encode the upgrade, downgrade. perform, etc.
    # a deeper dive is to load historical updates and look for stocks, reviewers, number of reviews OR size of upgrade that leads to a price change.
    # Process the new recommendations:
    #for rec in new_recs_map.items():
    #    print(rec)

    if debug:
        infos = ["Alcoa upgraded to Overweight from Equal Weight at Morgan Stanley",
         "Morgan Stanley upgrades Alcoa to Overweight on better prospects for aluminum",
         "GameStop downgraded to Hold from Buy at Jefferies",
    "GameStop upgraded to Outperform from Market Perform at Telsey Advisory",
    "GameStop upgraded to Buy ahead of video game console cycle at Jefferies",
    "Alcoa upgraded to Buy with aluminum prices rising at Deutsche Bank",
    "Alcoa upgraded to Buy from Hold at Deutsche Bank",
    "Hedgeye adds Palantir to best idea short list, Bloomberg says",
    "Palantir assumed with an Underperform at Credit Suisse",
    "Palantir downgraded to Underperform from Market Perform at William Blair",
    "Palantir downgraded to Sell from Neutral at Citi",
    "Credit Suisse downgrades 'disconnected from fundamentals' Palantir to sell",# »# seems like a duplicate
    "Palantir downgraded to Underweight from Equal Weight at Morgan Stanley",
    "Alcoa upgraded to Buy from Hold at Deutsche Bank",
    'Peter Thiel reports 6.6% passive stake in Palantir',
    'Senvest Management reports 5.54% passive stake in GameStop',
                 'Citron shorting Palantir, sees $20 stock by end of 2020',
                 'Amazon.com assumed with an Outperform at Wolfe Research',
        'Alcoa initiated with a Sell at Goldman Sachs',
        'Palantir initiated with a Market Perform at William Blair',
                 'Alcoa options imply 10.0% move in share price post-earnings',
        'GameStop believes it has sufficient liquidity to fund operations',
        'GameStop says \'Reboot\' is delivering lower costs, reduced debt',
        'Palantir, Rio Tinto sign multi-year enterprise partnership',
        'PG&E begins deployment of Palantir\'s Foundry Software',
        'Citi says sell Palantir on deceleration in growth, upcoming lockup expiry',
        'Pentagon cybersecurity project slowed by flaws, Bloomberg says',
        'Palantir awarded contract from Army worth up to $250M',
        'Palantir awarded $44.4M contract from FDA',
        'Fujitsu signs $8M contract as Palantir Foundry customer',
        'Army Vantage reaffirms Palantir partnership with $114M agreement',
        'Palantir provides update on partnership with Greece to support COVID-19 response',
        'Palantir receives Army prototype contract to support network modernization',
        'Soros takes stake in Palantir, exits TransDigm position',
        "AOC flags \'material risks\' to Palantir investors in SEC letter, TechCrunch says",
        "PetIQ \'a smart way\' to play pet care boom, Barron's says",]
        for i in infos:
            print('')
            print(i)
            s=Sentiment()
            s.Parse(i,sid=sid,nlp=nlp)
            print(s)
