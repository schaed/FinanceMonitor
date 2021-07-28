import os,sys
import datetime
import pandas as pd
import numpy as np
import base as b
import math
import datetime,time

# Collect info from finviz
def collect(URLin = 'https://finviz.com/screener.ashx?v=340\&s=ta_topgainers\&r=',total_table=[],maxIndex=2,name=''):
    """collect - 
    input: URL - str - path
           total_table - DataFrame - inputs
           maxIndex - int - max number of pages 10 per page
           """
    for i in range(0,maxIndex):
        URL = URLin+'%s' %(1+i*20)
        filename_rec='/tmp/topgain%i.html' %i
        if not os.path.exists(filename_rec):
            os.system('wget -T 30 -q -O %s %s' %(filename_rec,URL))
            
        table_MN = pd.read_html(filename_rec)
        ticker=''
        os.system('rm %s' %filename_rec)
        for t in table_MN:
            #print(t.columns)
            #print('  ')
            for j in t.index:
                my_list = t.loc[j].tolist()
                if len(my_list)>0 and my_list[0]=='Ticker':
                    #print(my_list[1])
                    ticker=my_list[1].split(' ')[0]
                    #print(ticker)
                if len(my_list)>0 and my_list[0]=='Market Cap':
                    #print(my_list[1])
                    #print(t)
                    #t.columns = t.loc[j].tolist()
                    #t=t.drop(0)
                    #print('')
                    vals = np.concatenate((t.loc[:,1].values, t.loc[:,3].values,np.array([ticker,name])), axis=None) 
                    cols = np.concatenate((t.loc[:,0].values, t.loc[:,2].values,['ticker','save_type']), axis=None) 
                    newT = pd.DataFrame([vals],columns=cols)
                    
                    for a in ['Market Cap','P/E','Forward P/E','Insider Own','Short Float','Analyst Recom','Avg Volume','Insider Trans','P/S','PEG','P/B','Inst Own','Inst Trans','Target Price']:
                        newT[a] = pd.to_numeric(newT[a].str.strip('%').replace({'K': 'e3', 'M': 'e6','B': 'e9','T': 'e12'}, regex=True),errors='coerce',downcast='integer')
                    #print(newT)
                    #print(newT.dtypes)
                    if len(total_table)==0:
                        total_table = newT
                    else:
                        # make sure we don't duplicate
                        for tick in  newT.ticker.values:
                            if tick in total_table.ticker.values:
                                #print(newT[newT.ticker==tick])
                                newT.drop(newT[newT.ticker==tick].index.values,inplace=True)
                            
                        total_table = pd.concat([total_table,newT])
    return total_table


if __name__ == "__main__":
    # execute only if run as a script
    today = datetime.datetime.today()
    total_table_top_gain=[]
    total_entries=0
    #total_table_unusualvolume=[]
    #total_table_top_loser=[]
    
    outFileName='News/table_%s_%s_%s.csv' %(today.day,today.month,today.year)
    while (today.hour<23 or (today.hour==23 and today.minute<30)):
        try:
            total_table_top_gain = collect(URLin = 'https://finviz.com/screener.ashx?v=320\&s=ta_unusualvolume\&r=',total_table=total_table_top_gain,maxIndex=2,name='unusualvolume')
        except:
            print('failed...volume')
            sys.stdout.flush()
        try:
            total_table_top_gain = collect(URLin = 'https://finviz.com/screener.ashx?v=340\&s=ta_topgainers\&r=',total_table=total_table_top_gain,maxIndex=2,name='top_gain')
        except:
            print('failed...top gain')
            sys.stdout.flush()
        try:
            total_table_top_gain = collect(URLin = 'https://finviz.com/screener.ashx?v=340\&s=ta_toplosers\&r=',total_table=total_table_top_gain,maxIndex=2,name='top_loser')
        except:
            print('failed...top losers')
            sys.stdout.flush()
        try:
            total_table_top_gain = collect(URLin = 'https://finviz.com/screener.ashx?v=210\&s=ta_overbought\&r=',total_table=total_table_top_gain,maxIndex=2,name='overbought')
        except:
            print('failed...over bought')
            sys.stdout.flush()
        print(total_table_top_gain)
        sys.stdout.flush()
        today = datetime.datetime.today()
        
        if len(total_table_top_gain)>total_entries:
            total_entries = len(total_table_top_gain)
            total_table_top_gain.to_csv(outFileName,index=False)
        time.sleep(300)

#ta_overbought
#ta_oversold
