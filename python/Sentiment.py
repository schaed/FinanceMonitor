from dataclasses import dataclass
import datetime,time,re
debug=False

def ReturnSpacyDoc(inputTxt, nlp):
    """ ReturnSpacyDoc - parse data. not currently functional
    
        Parameters:
         inputTxt - str
              Text to be parsed
         nlp - spacey parser
    """
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
    """ FilterResult - removes words or strings from a list
    
        Parameters:
         filter_list - array of str
              Text to be filtered
         my_list - array of str
            Text to be filtered
    """
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
    """Store the decision, the type as well as other info

        Parameters:
             message:           str   = None # 'options' for put or call, target for a price target, position for announcing position, recom = recommendation, 'stake'=has stock, 'news' just trying to say positive or negative, 'clinicaltrial': is report on clinical trial
             ticker:            str   = None # ticker symbol
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

    """
    message:           str   = None # 'options' for put or call, target for a price target, position for announcing position, recom = recommendation, 'stake'=has stock, 'news' just trying to say positive or negative, 'clinicaltrial': is report on clinical trial
    ticker:            str   = None # ticker symbol
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

    # check the combo of different assessments.
    # hand it a list of options to determine the ranking.
    # lower text is the text that we are parsing
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

    def ParseEarnings(self,inputTxt, companyName='', ticker='', nlp=None, doc=None):
        """ParseEarnings - Parse the text to fill attributes

            inputTxt - str - news story text
            companyName - str - company name
            ticker - str - ticker symbol
            nlp - Spacey NLP
        """
        #TODO Capital Bancorp reports Q1 EPS 65c, consensus 56c
        #TODO BankUnited reports Q1 EPS $1.06, consensus 74c
        #TODO MarineMax raises FY21 EPS view to $5.50-$5.65 from $4.00-$4.20, consensus $4.35
        #TODO Sandy Spring Bancorp reports Q1 EPS $1.58, consensus $1.0
        #TODO Cleveland-Cliffs sees FY21 adjusted EBITDA $4B
        #TODO Lennar reports Q2 adjusted EPS $2.95, consensus $2.36
        self.message='earnings'
        if (re.search('reports',     inputTxt, re.IGNORECASE) and re.search('EPS',     inputTxt, re.IGNORECASE) and (re.search('consensus',     inputTxt, re.IGNORECASE) or re.search('last year',     inputTxt, re.IGNORECASE))):
            self.message='earnings'
            money = [ent for ent in doc.ents if ent.label_ == "MONEY"]
            for curr in ['EUR ','CHF ',' GBp','c']:
                if re.search(curr,   inputTxt, re.IGNORECASE):
                    money = [tok for tok in doc if tok.pos_ == "NUM"];
                    if curr=="c":
                        money = [tok for tok in doc if (tok.pos_ == "NUM") or (tok.pos_ != "NUM" and str(tok).strip('c').isdigit())];
                    break;
            imon=0

            for mon in money:
                if re.search('\('+str(mon)+'\)',   inputTxt, re.IGNORECASE):
                    money[imon] = '-'+str(mon).strip().strip('c')
                try:
                    if str(mon).count('c'):
                        money[imon] = str(mon).strip().strip('c')
                        money[imon] = float(money[imon])/100.0
                except:
                    print('Failed to convert')
                imon+=1
            if len(money)==1:
                self.price_after = money[0]
                #if 'initiat' in inputTxtLower:
            if len(money)==2:
                self.price_after  = money[0]
                self.price_before = money[1]

            if len(money)==0:
                icurr=0
                for w in inputTxt.split(' '):
                    for curr in ['c']:
                        if w.count(curr):
                            if (w.strip('c,')).isdigit():
                                if icurr==0: self.price_after  = float(w.strip('c,'))/100.0
                                if icurr==1: self.price_before  = float(w.strip('c,'))/100.0
                                icurr+=1
        return
    
    # handed a short string to determine meaning
    def Parse(self, inputTxt, companyName='', ticker='', sid=None, nlp=None, is_earnings=False):
        """Parse - Parse the text to fill attributes

            inputTxt - str - news story text
            companyName - str - company name
            ticker - str - ticker symbol
            sid - Spacey vader sentiment, but could be changed to another analyzer
            nlp - Spacey NLP
            is_earnings - Bool - stating if this is from an earnings webpage to facilitate the parsing
        """
        doc = nlp(inputTxt)
        inputTxtLower = inputTxt.lower()
        self.ticker = ticker
        # start with a rough vader analysis. Not sure it is has the right message in this context
        # returns a score: {'neg': 0.686, 'neu': 0.314, 'pos': 0.0, 'compound': -0.6572} 
        polscore = sid.polarity_scores(inputTxt)
        if debug:
            for k in sorted(polscore): print('{0}: {1}, '.format(k, polscore[k]), end='')
        self.vader_neg,self.vader_pos,self.vader_neu,self.vader_compound = polscore['neg'],polscore['pos'],polscore['neu'],polscore['compound']

        
        if is_earnings:
            self.ParseEarnings(inputTxt, companyName, ticker, nlp, doc)
            return
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

        # IPO
        #WalkMe indicated to open at $33.20, IPO priced at $31
        if re.search('IPO priced at',   inputTxt, re.IGNORECASE):
            self.message='ipo'
            money = [tok for tok in doc if tok.pos_ == "NUM"];
            if len(money)==1:
                self.price_after = money[0]
            if len(money)==2:
                self.price_after  = money[0]
                self.price_before = money[1]
                
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
        describ1 = ['underperform','sell','underweight','negative','reduce','bearish']
        describ2 = ['equal weight','market perform','neutral','hold','mixed','peer perform']
        describ3 = ['outperform','overweight','buy','positive','add','bullish']
        for init in ['reiterate','restate','re-iterate','re-state','transferred with']:
            if re.search(init, inputTxt, re.IGNORECASE): self.sentiment=0; self.message='recom'
            for de in describ3:
                if re.search(de, inputTxt, re.IGNORECASE): self.sentiment=1;
        # not sure of the message, but checking for the following for a hint: 
        for init in [' initiat',' re-initiat',' reinitiat',' assum',' re-assum',' reassum', 'resumed with a']:
            if re.search(init,  inputTxt, re.IGNORECASE): self.first_rev=True; self.message='recom'
        if re.search('upgrade',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'
        if re.search(' elevate',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'
        #if re.search(' rating',   inputTxt, re.IGNORECASE): self.sentiment=1; self.message='recom'
        if re.search('downgrade', inputTxt, re.IGNORECASE): self.sentiment=-1; self.message='recom'
        if re.search('lower',     inputTxt, re.IGNORECASE): self.sentiment=-1; #self.message='recom'
        if re.search(' rais',      inputTxt, re.IGNORECASE): self.sentiment=1; #self.message='recom'
        if re.search('resumed with a buy',      inputTxt, re.IGNORECASE): self.sentiment=1; #self.message='recom'
        if re.search('resumed with a hold',      inputTxt, re.IGNORECASE): self.sentiment=0; #self.message='recom'
        if re.search('resumed with a sell',      inputTxt, re.IGNORECASE): self.sentiment=-1; #self.message='recom'

        # Review levels. Search for each review level
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
        if self.message==None and (re.search('clinical trial',     inputTxt, re.IGNORECASE) or (re.search('trial',inputTxt, re.IGNORECASE) and re.search('phase',inputTxt, re.IGNORECASE)) or re.search('Phase 2a study',inputTxt, re.IGNORECASE)):
            self.message='clinicaltrial'
            # status
            if re.search('complete',     inputTxt, re.IGNORECASE):
                self.status = 'complete'
                self.sentiment = 1
            if re.search('preliminary',     inputTxt, re.IGNORECASE):
                self.status = 'preliminary'
                self.sentiment = 0.75
            for stat in ['start','launch','begin','initiate']:
                if re.search('start',     inputTxt, re.IGNORECASE):
                    self.status = 'start'
                    self.sentiment = 0.5
            # phase
            if re.search('Phase 1a',     inputTxt, re.IGNORECASE): self.phase = '1a'
            if re.search('Phase 1b',     inputTxt, re.IGNORECASE): self.phase = '1b'
            if re.search('Phase 3',     inputTxt, re.IGNORECASE): self.phase = '3'
            if re.search('Phase 2',     inputTxt, re.IGNORECASE): self.phase = '2'
            if re.search('Phase 2a',     inputTxt, re.IGNORECASE): self.phase = '2a'

        #TODO Capital Bancorp reports Q1 EPS 65c, consensus 56c
        #TODO BankUnited reports Q1 EPS $1.06, consensus 74c
        #TODO MarineMax raises FY21 EPS view to $5.50-$5.65 from $4.00-$4.20, consensus $4.35
        #TODO Sandy Spring Bancorp reports Q1 EPS $1.58, consensus $1.0
        #TODO Cleveland-Cliffs sees FY21 adjusted EBITDA $4B
        if self.message==None and (re.search('reports',     inputTxt, re.IGNORECASE) and re.search('EPS',     inputTxt, re.IGNORECASE) and re.search('consensus',     inputTxt, re.IGNORECASE)):
            self.ParseEarnings(inputTxt, companyName, ticker, nlp, doc)
            return
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

    # convert the tokens to a digit
    #def Convert
    
    # processing the price target
    def PassPriceTarget(self):
        if self.message=='target':
            price_after  = str(self.price_after).replace(',','')
            price_before = str(self.price_before).replace(',','')
            print('analyzing price target on %s with price after %s' %(self.ticker,price_after))
            if (price_after.replace('.','',1)).isdigit() and (price_before.replace('.','',1)).isdigit():
                price_after = float(price_after)
                price_before = float(price_before)
                if price_after!=-1 and price_before!=-1:
                    if (price_after/(price_before+0.00001))>1.07 and (price_after-price_before)>0.07:
                        return [price_after,price_before]
        return []
    # processing the earnings
    def PassEarnings(self):
        if self.message=='earnings':
            print('analyzing price target on %s with price after %s' %(self.ticker,price_after))
            price_after  = str(self.price_after).replace(',','')
            price_before = str(self.price_before).replace(',','')
            print('analyzing price target on %s with price after %s' %(self.ticker,price_after))
            if (price_after.replace('.','',1)).isdigit() and (price_before.replace('.','',1)).isdigit():
                price_after = float(price_after)
                price_before = float(price_before)
                if price_after!=-1 and price_before!=-1:
                    if (price_after/(price_before+0.00001))>1.07 and (price_after-price_before)>0.07:
                        return [price_after,price_before]
        return []
    
    # processing the pharma
    def PharmaPhase(self):
        if self.message=='clinicaltrial':
            print('Clinical trial on %s' %self.ticker)
            return True
        return False
    
    # processing the upgrade of company
    def PassUpgrade(self):
        if self.message=='recom':
            print('analyzing this recommendation on %s' %self.ticker)
            if self.assessment_result>1 and self.assessment_change>0:
                return True
        return False

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
    is_earnings: bool  = False # true if reporting earnings
    
    # process request to set the sentiment
    def Sentiment(self,inputTxt=None, sid=None, nlp=None, is_earnings=False):
        """Sentiment - create a sentiment object and parse it

            inputTxt - str - news story text
            sid - Spacey vader sentiment, but could be changed to another analyzer
            nlp - Spacey NLP
            is_earnings - Bool - stating if this is from an earnings webpage to facilitate the parsing
        """
        if inputTxt==None: inputTxt = self.shortText
        ticker=''
        if len(self.tickers)>0:
            ticker = self.tickers[0]
        if debug:
            print('')
            print('Input: %s'%inputTxt)
        sent = Sentiment()
        sent.Parse(inputTxt,self.company, ticker, sid=sid, nlp=nlp, is_earnings=is_earnings)
        return sent

    # collect price info from before
    def ProcessSentiment(self, sentiment, api=None):
        """ProcessSentiment - make a buy or sell decision. not functional

            sentiment - Sentiment object with data extracted
            api - alpaca API to read in data if needed
        """
        ticker=''
        if len(self.tickers)>0:
            ticker = self.tickers[0]
            
        if ticker!='':
            # read in the daily data. not adjusted. collecting from alpaca
            stock_info = runTicker(api,ticker)
            print(stock_info)
            sentiment.Process() #ticker
