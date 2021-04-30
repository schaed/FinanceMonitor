import pandas as pd
import re,os,sys
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import datetime,time
import wget,pickle
from dataclasses import dataclass
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

def ReturnSpacyDoc(inputTxt):
    Token.set_extension("is_rating", default=False)
    doc = nlp("I like David Bowie")
    ## Add attribute ruler with exception for "The Who" as NNP/PROPN NNP/PROPN
    #ruler = nlp.get_pipe("attribute_ruler")
    ## Pattern to match "The Who"
    #patterns = [[{"LOWER": "the"}, {"TEXT": "Who"}]]
    ## The attributes to assign to the matched token
    #attrs = {"TAG": "NNP", "POS": "PROPN"}
    ## Add rules to the attribute ruler
    #ruler.add(patterns=patterns, attrs=attrs, index=0)  # "The" in "The Who"
    #ruler.add(patterns=patterns, attrs=attrs, index=1)  # "Who" in "The Who"
            #print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
        #print(":", [ent.text for ent in doc.ents])
        #print(":", [ent.label_ for ent in doc.ents])
        #print(":", [ent for ent in doc.ents if ent.label_ == "ORG"]) #Companies, agencies, institutions.
        #print(":", [ent for ent in doc.ents if ent.label_ == "PERSON"])
        #print(":", [ent for ent in doc.ents if ent.label_ == "GPE"]) #Geopolitical entity, i.e. countries, cities, states.
        #print(":", [ent for ent in doc.ents if ent.label_ == "MONEY"]) #Monetary values, including unit.
        #print(":", [ent for ent in doc.ents if ent.label_ == "NNP"])
    return doc

# This filters out a list of elements
def FilterResult(filter_list, my_list):
    buildNewList=[]
    for m in my_list:
        foundTxt=False
        for f in filter_list:
            if debug: print(f)
            if debug: print(m)
            if re.search( str(f), str(m), re.IGNORECASE):
                foundTxt=True
                break

        if not foundTxt: buildNewList+=[m]
    return buildNewList

@dataclass
class Sentiment:
    """"Store the decision, the type as well as other info
    """
    message:           str   = None # 'options' for put or call, target for a price target, position for announcing position, recom = recommendation, 'stake'=has stock, 'news' just trying to say positive or negative, 'clinicaltrial': is report on clinical trial
    sentiment:         float = None # -1 downgrade, 0 neutral, 1 is upgrade.
    assessment_result: int   = None # 0 underweight, 2 neutral, 3 is overweight.
    assessment_change: int   = None # Delta or change. Positive is an upgrade and negative is downgrade
    assessment_stake:  float = None # passive stake notes in percentage
    assessment_amt:    float = None # amount for grants or other things
    price_before:      float = -1 # previous price target
    price_after:       float = -1 # after price target
    assessment_by:     str   = 'unknown' # company
    assessment_opt:    int   = None # is this an options assement? -1 for bearish, 0 unsure, and 1 for bullish
    assessment_vol:    float = None # is this an options assement? options implied volatility
    assessment_consistent: bool = True
    first_rev:         bool  = False
    vader_pos:         float  = None
    vader_compound:    float  = None
    vader_neg:         float  = None
    vader_neu:         float  = None
    phase:             str    = None # clinical trial phase
    status:            str    = None # clinical status

    # check the combo.
    def CheckCombo(self, describ, inputTxtLower):

        if self.assessment_result !=None:
            return self.assessment_result
        inputTxtLowerSplit = inputTxtLower.split('to')
        if debug: print(inputTxtLowerSplit)
        j1=0
        j2=0
        #from Underperform to Market
        if len(inputTxtLowerSplit)>1:
            foundLvl=False
            for level in describ:
                for word in level:
                    if re.search(word,  inputTxtLowerSplit[0], re.IGNORECASE):
                        foundLvl=True; break
                if foundLvl: break
                j1+=1
            foundLvl=False
            for level in describ:
                for word in level:
                    if re.search(word,  inputTxtLowerSplit[1], re.IGNORECASE):
                        foundLvl=True; break
                if foundLvl: break
                j2+=1
        if j1>=len(describ) or j2>=len(describ):

            inputTxtLowerSplit = inputTxtLower.split('from')
            if debug: print(inputTxtLowerSplit)
            j1=0
            j2=0
            #to Underperform from Market
            if len(inputTxtLowerSplit)>1:
                foundLvl=False
                for level in describ:
                    for word in level:
                        if re.search(word,  inputTxtLowerSplit[0], re.IGNORECASE):
                            foundLvl=True; break
                    if foundLvl: break
                    j1+=1
                foundLvl=False
                for level in describ:
                    for word in level:
                        if re.search(word,  inputTxtLowerSplit[1], re.IGNORECASE):
                            foundLvl=True; break
                    if foundLvl: break
                    j2+=1
        else: # deals with from X to Y
            self.assessment_change = j2 - j1
            self.assessment_result = j2
            return self.assessment_result
        if j1>=len(describ) or j2>=len(describ):
            if debug: print('failure to understand how the ranking changed: %s' %inputTxtLower)
        else: # deals with to X from Y
            self.assessment_change = j1 - j2
            self.assessment_result = j1
        return self.assessment_result

    # handed a short string to determine meaning
    def Parse(self, inputTxt, companyName=''):
        doc = nlp(inputTxt)
        inputTxtLower = inputTxt.lower()

        # start with a rough vader analysis. Not sure it is has the right message in this context
        # returns a score: {'neg': 0.686, 'neu': 0.314, 'pos': 0.0, 'compound': -0.6572} 
        polscore = sid.polarity_scores(inputTxt)
        if debug:
            for k in sorted(polscore): print('{0}: {1}, '.format(k, polscore[k]), end='')
        self.vader_neg,self.vader_pos,self.vader_neu,self.vader_compound = polscore['neg'],polscore['pos'],polscore['neu'],polscore['compound']
        # This is an options statement
        # Alcoa call volume above normal and directionally bullish »
        # Alcoa put volume heavy and directionally bearish »
        # Palentir Technologies call volume picks up after quiet week »
        if re.search('call volume above normal',   inputTxt, re.IGNORECASE) or re.search('call volume heavy',   inputTxt, re.IGNORECASE) or re.search('heavy call volume',   inputTxt, re.IGNORECASE):
            self.assessment_opt=1; self.message='options'; 
        if re.search('put volume above normal',   inputTxt, re.IGNORECASE) or re.search('put volume heavy',   inputTxt, re.IGNORECASE) or re.search('heavy put volume',   inputTxt, re.IGNORECASE):
            self.assessment_opt=-1; self.message='options'

        # Alcoa options imply 10.0% move in share price post-earnings »
        # GameStop options imply 32.5% move in share price post-earnings »
        if re.search('options imply',   inputTxt, re.IGNORECASE):
            self.message='options'; self.assessment_opt=0;
            money = [tok for tok in doc if tok.pos_ == "NUM"];
            if len(money)>0:
                self.assessment_vol =  money[0]

        # price target
        # Alcoa price target raised to $20 from $14 at B. Riley Securities »
        # Alcoa price target raised to $38 from $36 at BMO Capital »
        # Alcoa price target lowered to $20 from $22 at B. Riley Securities »
        # Alcoa price target raised to $22 from $20 at B. Riley Securities »
        if re.search('price target',   inputTxtLower, re.IGNORECASE):
            self.message='target'
            money = [ent for ent in doc.ents if ent.label_ == "MONEY"]
            for curr in ['EUR ','CHF ',' GBp']:
                if re.search(curr,   inputTxtLower, re.IGNORECASE):
                    money = [tok for tok in doc if tok.pos_ == "NUM"];
                    break;
                    
            if len(money)==1:
                self.price_after = money[0]
                #if 'initiat' in inputTxtLower:
            if len(money)==2:
                self.price_after  = money[0]
                self.price_before = money[1]
                if 'rais'  in inputTxtLower:                       self.sentiment=1
                if 'lower' in inputTxtLower:                       self.sentiment=-1                    
                if 'rais'  in inputTxtLower and self.price_after<self.price_before: self.assessment_consistent = False
                if 'lower' in inputTxtLower and self.price_after>self.price_before: self.assessment_consistent = False

        # special case for passive stake
        # Senvest Management reports 5.54% passive stake in GameStop 
        # Peter Thiel reports 6.6% passive stake in Palantir »
        # Soros takes stake in Palantir, exits TransDigm position
        if re.search('passive stake',   inputTxtLower, re.IGNORECASE) or re.search('takes stake',   inputTxtLower, re.IGNORECASE) or re.search('buys stake',   inputTxtLower, re.IGNORECASE) or re.search('a buyer of',   inputTxtLower, re.IGNORECASE):
            self.message = 'stake'
            money = [tok for tok in doc if tok.pos_ == "NUM"];
            if len(money)>0:
                self.assessment_stake = money[0]
            #Piper Sandler a buyer of Align Technology ahead of Q1 results
            if re.search('a buyer of',   inputTxtLower, re.IGNORECASE):
                self.sentiment = 1
                
        # special case for shorting example.
        # Citron shorting Palantir, sees $20 stock by end of 2020 »
        if re.search('shorting',   inputTxtLower, re.IGNORECASE):
            self.message = 'recom'
            self.sentiment = -1
            money = [ent for ent in doc.ents if ent.label_ == "MONEY"]
            if len(money)>0: self.price_after = money[0]

        # random idea messages
        #Palantir falls after Citron announces short position with $20 target
        #GameStop added to long side of Investing Ideas list at Hedgeye
        #GameStop added to short side of Investing Ideas list at Hedgeye
        # Hedgeye adds Palantir to best idea short list, Bloomberg says
        if (re.search('short position',   inputTxtLower, re.IGNORECASE) or re.search('long position',   inputTxtLower, re.IGNORECASE) or
            re.search('short side',   inputTxtLower, re.IGNORECASE) or re.search('long side',   inputTxtLower, re.IGNORECASE) or
            re.search('short list',   inputTxtLower, re.IGNORECASE) or re.search('long list',   inputTxtLower, re.IGNORECASE)):
            self.message='position'
            if re.search('short ',   inputTxtLower, re.IGNORECASE): self.sentiment=-1
            if re.search('long ',   inputTxtLower, re.IGNORECASE): self.sentiment=1
            if re.search(' add',   inputTxtLower, re.IGNORECASE) and re.search('short ',   inputTxtLower, re.IGNORECASE): self.sentiment=-1
            if re.search('remov ',   inputTxtLower, re.IGNORECASE) and re.search('short ',   inputTxtLower, re.IGNORECASE): self.sentiment=1
            if re.search(' add',   inputTxtLower, re.IGNORECASE) and re.search('long ',   inputTxtLower, re.IGNORECASE): self.sentiment=1
            if re.search(' remov',   inputTxtLower, re.IGNORECASE) and re.search('long ',   inputTxtLower, re.IGNORECASE): self.sentiment=-1
            if re.search('target',   inputTxtLower, re.IGNORECASE):
                self.message='target'
                money = [ent for ent in doc.ents if ent.label_ == "MONEY"]
                if len(money)==1:
                    self.price_after = money[0]
                if len(money)==2:
                    self.price_after = money[0]
                    self.price_before = money[1]
                
        # This is a review..
        #Alcoa upgraded to Overweight from Equal Weight at Morgan Stanley »
        #Credit Suisse downgrades 'disconnected from fundamentals' Palantir to sell »# seems like a duplicate
        #Palantir downgraded to Underweight from Equal Weight at Morgan Stanley »
        #Alcoa upgraded to Buy from Hold at Deutsche Bank »
        # way to say this is a new review
        #Amazon.com assumed with an Outperform at Wolfe Research
        #Alcoa initiated with a Sell at Goldman Sachs »
        #Palantir initiated with a Market Perform at William Blair »
        # I don't know whether positive for negative, but this is a refresh of the statement.
        # TODO: Grainger elevated to bullish Fresh Pick at Baird
        # TODO: Citi keeps Sell rating, $159 target on Tesla into Q1 results
        # TODO: Citi keeps Sell rating, $159 target on Tesla into Q1 results
        for init in ['reiterate','restate','re-iterate','re-state']:
            if re.search(init, inputTxt, re.IGNORECASE): self.sentiment=0; self.message='recom'
        # not sure of the message, but checking for the following for a hint: 
        for init in [' initiat',' re-initiat',' reinitiat',' assum',' re-assum',' reassum']:
            if re.search(init,  inputTxt, re.IGNORECASE): self.first_rev=True; self.message='recom'
        if re.search('upgrade',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'
        if re.search(' elevate',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'        
        #if re.search(' rating',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'        
        if re.search('downgrade', inputTxt, re.IGNORECASE): self.sentiment=-1; self.message='recom'
        if re.search('lower',     inputTxt, re.IGNORECASE): self.sentiment=-1; #self.message='recom'
        if re.search(' rais',      inputTxt, re.IGNORECASE): self.sentiment=1; #self.message='recom'

        # Review levels. Search for each review level
        describ1 = ['underperform','sell','underweight','negative','reduce','bearish']
        describ2 = ['equal weight','market perform','neutral','hold','mixed','peer perform']
        describ3 = ['outperform','overweight','buy','positive','add','bullish']
        if self.sentiment==None:
            if re.search('bullish', inputTxt, re.IGNORECASE):   self.sentiment=1
            if re.search('bearish', inputTxt, re.IGNORECASE):   self.sentiment=-1
            if re.search('neutral', inputTxt, re.IGNORECASE):   self.sentiment=0
        else:
            self.assessment_result = self.CheckCombo([describ1,describ2,describ3], inputTxtLower)
                
        if self.first_rev or (self.sentiment!=None and self.assessment_result==None):
            for d in describ1:
                if re.search(d,  inputTxt, re.IGNORECASE):
                    self.assessment_result=0
                    break
            for d in describ2:
                if re.search(d,  inputTxt, re.IGNORECASE):
                    self.assessment_result=1
                    break
            for d in describ3:
                if re.search(d,  inputTxt, re.IGNORECASE):
                    self.assessment_result=2
            # If we can infer the change, then let's set it. If it says upgrade or downgrade, then let's apply it.
            if self.assessment_change == None:
                if re.search('upgrade',     inputTxt, re.IGNORECASE): self.assessment_change = 1
                if re.search('downgrade',   inputTxt, re.IGNORECASE): self.assessment_change = -1

        # UNDER clinicaltrial could add reading for clinical trials...maybe own category
        # TODO::; CohBar completes last subject visit in Phase 1b clinical trial for CB4211
        if self.message==None and re.search('clinical trial',     inputTxt, re.IGNORECASE):
            self.message='clinicaltrial'
            # status
            if re.search('complete',     inputTxt, re.IGNORECASE):
                self.status = 'complete'
                self.sentiment = 1
            for stat in ['start','launch','begin','initiate']:
                if re.search('start',     inputTxt, re.IGNORECASE):
                    self.status = 'start'
                    self.sentiment = 0.5
            # phase
            if re.search('Phase 1b',     inputTxt, re.IGNORECASE): self.phase = '1b'

        #TODO Capital Bancorp reports Q1 EPS 65c, consensus 56c
        #TODO BankUnited reports Q1 EPS $1.06, consensus 74c
        #TODO MarineMax raises FY21 EPS view to $5.50-$5.65 from $4.00-$4.20, consensus $4.35
        #TODO Sandy Spring Bancorp reports Q1 EPS $1.58, consensus $1.0
        #TODO Cleveland-Cliffs sees FY21 adjusted EBITDA $4B
        if self.message==None and (re.search('reports',     inputTxt, re.IGNORECASE) and re.search('EPS',     inputTxt, re.IGNORECASE) and re.search('consensus',     inputTxt, re.IGNORECASE)):
            self.message='earnings'
        # extra options. Just trying to understand positive versus negative
        # GameStop believes it has sufficient liquidity to fund operations
        # GameStop says 'Reboot' is delivering lower costs, reduced debt
        # Palantir, Rio Tinto sign multi-year enterprise partnership
        # PG&E begins deployment of Palantir's Foundry Software
        # Citi says sell Palantir on deceleration in growth, upcoming lockup expiry
        # Pentagon cybersecurity project slowed by flaws, Bloomberg says
        # Palantir awarded contract from Army worth up to $250M
        # Palantir awarded $44.4M contract from FDA
        # Fujitsu signs $8M contract as Palantir Foundry customer
        # Army Vantage reaffirms Palantir partnership with $114M agreement
        # Palantir provides update on partnership with Greece to support COVID-19 response
        # Palantir receives Army prototype contract to support network modernization
        # AOC flags 'material risks' to Palantir investors in SEC letter, TechCrunch says
        # PetIQ 'a smart way' to play pet care boom, Barron's says
        if self.message==None:
            self.message='news'
            #self.sentiment=0
            #now just gonna start with a max, and set the score
            allpolscores = [polscore['neg'],polscore['pos'],polscore['neu']]
            if polscore['pos'] == max(allpolscores): self.sentiment = polscore['pos']            
            elif polscore['neg'] == max(allpolscores): self.sentiment = -1*polscore['neg']
            elif polscore['neu'] == max(allpolscores): self.sentiment = 0
            # Run some corrections by looking for groups of words
            neg_groups = [['sees risk','flag','flagged','slow','sell','bad sign','dark sign','falls'],['risk','concern','debt','flaw','deceleration']]
            pos_groups = [['reaffirm','signs',' sign','award',' grant',' order','begin','rises','deliver','has','acceleration','receives'],['partner','deal','contract','deploy','low cost','lower cost','low debt','lower debt','sufficient liquidity']]
            neg_count=0
            pos_count=0
            for negg in neg_groups:
                for nphrase in negg:
                    if re.search(nphrase,     inputTxt, re.IGNORECASE): neg_count+=1; break
            for posg in pos_groups:
                for pphrase in posg:
                    if re.search(pphrase,     inputTxt, re.IGNORECASE): pos_count+=1; break                
            if neg_count>1 and pos_count>1: pass
            elif neg_count<2 and pos_count>1: self.sentiment = 1
            elif neg_count>1 and pos_count<2: self.sentiment = -1
                
            
            # decide on the amount if it exists
            money = [ent for ent in doc.ents if ent.label_ == "MONEY"]
            if len(money)==1:
                self.assessment_amt = money[0]
            else:
                money = [tok for tok in doc if tok.pos_ == "NUM"];
                if len(money)>0:
                    self.assessment_stake = money[0]

        # collect who is reporting this
        if inputTxtLower.find('at'):
            mytext = inputTxt.lower().split(' at ')
            if len(mytext)>1:
                self.assessment_by=mytext[1].strip()
                shorterText = self.assessment_by.strip().split(',')
                if self.assessment_by.count(',') and len(shorterText)>0:
                    self.assessment_by=shorterText[0].strip()
        elif re.search('from',     inputTxt, re.IGNORECASE):
            mytext = inputTxt.lower().split(' from ')
            if len(mytext)>0:
                self.assessment_by=mytext[0].strip()
                shorterText = self.assessment_by.split(',')
                if self.assessment_by.count(',') and len(shorterText)>0:
                    self.assessment_by=shorterText[0].strip()

        # special corrections for reviewers. trying to decipher the reviewer in more complicated cases. Some are hardcoded
        if self.assessment_by=='unknown':
            for except_fund in ['hedgeye','jefferies','citron','airforce','navy ','army ','aoc ','white house','pentagon','wedbush']:
                if re.search(except_fund,     inputTxt, re.IGNORECASE): self.assessment_by=except_fund.strip(' ')
            # still not sure who made this? Then let's try to reason it out with countries and proper nouns
            if self.assessment_by=='unknown':
                filter_list=describ1+describ2+describ3    # these sometimes show up
                if companyName!='': filter_list+=[companyName] # filter out this company name
                countries = [ent for ent in doc.ents if ent.label_ == "GPE"]
                companies = [ent for ent in doc.ents if ent.label_ == "ORG"]
                persons   = [ent for ent in doc.ents if ent.label_ == "PERSON"]
                countries = FilterResult(filter_list, countries)
                companies = FilterResult(filter_list, companies)
                persons   = FilterResult(filter_list, persons)
                if len(persons)>0: 
                    self.assessment_by = str(persons[0]).lower()                   
                elif len(companies)>0:
                    self.assessment_by = str(companies[0]).lower()
                elif len(countries)>0:
                    self.assessment_by = str(countries[0]).lower()
        # Just a little cleaning
        #if self.assessment_by!='unknown':
        #    if self.assessment_by.find('\'s'): self.assessment_by=self.assessment_by[:self.assessment_by.find('\'s')]

    # processing the sentiment analysis
    def Process(self):
        if self.message=='recom':
            print('analyzing this recommendation')
        elif self.message=='target':
            print('analyzing price target')
            
        else:
            print('cannot yet process this request')
        return

@dataclass
class News:
    """"Store the news with tickers, timeSlot: posting time, complete story and headline, company, current price of the company. 
    Previous stories are saved for lookup
    """
    tickers: tuple =()
    timeSlot: datetime.datetime = datetime.datetime.today()
    completeText: str = ''
    shortText: str = ''
    company: str = ''
    currPrice: float = 0
    upgradeRel: tuple = ()
    downgradeRel: tuple = ()
    nochangeRel: tuple = ()
    
    # process request to set the sentiment
    def Sentiment(self,inputTxt=None):
        if inputTxt==None: inputTxt = self.shortText
        if debug:
            print('')
            print('Input: %s'%inputTxt)
        sent = Sentiment()
        sent.Parse(inputTxt,self.company)
        return sent

    # collect price info from before
    def ProcessSentiment(self, sentiment):
        ticker=''
        if len(self.tickers)>0:
            ticker = tickers[0]
            
        if ticker!='':
            # read in the daily data. not adjusted. collecting from alpaca
            stock_info = runTicker(api,ticker)
            print(stock_info)
            sentiment.Process() #ticker


# Read the php or html. Then check for the short title as the key. build a struct.
def ParseTheFly(inputFileName='/tmp/recommend.php',my_map={},new_map={}):

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
                 shortText,company,currPrice,upgradeRel,downgradeRel,nochangeRel)

            # only process new stories
            if shortText not in my_map:
                my_map[shortText] = my_news
                new_map[shortText] = my_news

                # running the sentiment analysis
                print('')
                print(shortText)
                print(my_news.Sentiment())                
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

def Execute(f1='/tmp/recommend.php', f2='/tmp/news.php',total_news_map={},total_recs_map={}, outFileName='News.p'):

    new_news_map ={}
    new_recs_map ={}
    # read these in and save to a pickle file.
    ParseTheFly(inputFileName=f1,my_map=total_recs_map,new_map=new_recs_map)
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
            #after the fact https://thefly.com/news.php?market_mover_filter=on&h=5
            #maybe others https://www.americanbankingnews.com/category/market-news/analyst-articles-us/page/17
            #maybe finviz.com            
            URL = 'https://thefly.com/news.php?analyst_recommendations=on&h=2'
            filename_rec = '/tmp/recommend.php'
            os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))    
            URL = 'https://thefly.com/news.php'
            filename_news = '/tmp/newsInfo.php' # https://thefly.com/news.php?earnings_filter=on&h=3 earnings
            os.system('wget -T 30 -q -O %s %s' %(filename_news,URL))
            # download the results
            Execute(f1=filename_rec, f2=filename_news,total_news_map=total_news_map,total_recs_map=total_recs_map,outFileName=outFileName)
        except:
            print('Error downloading pages!')
        
        # sleep for 10 minutes
        time.sleep(600)
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
            s.Parse(i)
            print(s)
