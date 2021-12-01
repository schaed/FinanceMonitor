from ReadData import SQL_CURSOR,ALPHA_TIMESERIES,ConfigTable
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
draw=False
doPDFs=False
debug=False
#s=SQL_CURSOR()
readType='full'
outdir='/tmp/'

def MakePlot(xaxis, yaxis, xname='Date',yname='Beta',saveName='', hlines=[],title='',doSupport=False,my_stock_info=None, doScatter=False,doBox=False):
    """ Generic plotting with option to show support lines
         
         Parameters:
         xaxis : numpy array
            Date of stock value
         yaxis : numpy array
            Closing stock value
         xname : str
            x-axis name
         yname : str
            y-axis name
         saveName : str
            Saved file name
         hlines : array of horizontal lines drawn in matplotlib
         title : str
            Title of plot
         doSupport : bool
            Request generation of support lines on the fly
         my_stock_info : pandas data frame of stock timing and adj_close
         doScatter : bool - draw a scatter plot
         doBox : bool - draw a box plot for unique x-values
     """
    # plotting
    plt.clf()
    ax7=None
    fig7=None
    if doScatter:
        plt.scatter(xaxis,yaxis)
    elif doBox: 
        fig7, ax7 = plt.subplots()
        d1=[]
        for m in np.unique(xaxis.values):
            d1+=[yaxis.loc[xaxis==m].dropna()]
        bp = ax7.boxplot(d1,whis=[5,95],showmeans=True,notch=True)
        ax7.grid(True)
        ax7.legend([bp['medians'][0], bp['means'][0]],['median','mean'],loc="upper left")
        plt.title(saveName.replace('_',' '))
    else:
        plt.plot(xaxis,yaxis)
    plt.gcf().autofmt_xdate()
    plt.ylabel(yname)
    plt.xlabel(xname)
    if title!="":
        plt.title(title, fontsize=30)
    for h in hlines:
        plt.axhline(y=h[0],color=h[1],linestyle=h[2]) #xmin=h[1], xmax=h[2],
    if doSupport:
        techindicators.supportLevels(my_stock_info)
    if draw and ax7!=None: fig7.show()
    elif draw: plt.show()
    if doPDFs and ax7!=None: fig7.savefig(outdir+'%s.pdf' %(saveName))
    elif doPDFs: plt.savefig(outdir+'%s.pdf' %(saveName))
    if ax7!=None: fig7.savefig(outdir+'%s.png' %(saveName))
    else: plt.savefig(outdir+'%s.png' %(saveName))
    if not draw: plt.close()
    plt.close()

s = SQL_CURSOR(db_name='stocksPerfHistory.db')
sc = s.cursor()
table_names = sc.execute("SELECT name from sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';").fetchall()

sqlcursor = SQL_CURSOR()
ts = ALPHA_TIMESERIES()
spy,j = ConfigTable('SPY', sqlcursor,ts,readType, hoursdelay=15)

for tname in table_names:
    print(tname[0])
    if tname[0].count('-'):
        continue
    stock=None
    try:
        stock = pd.read_sql('SELECT * FROM %s' %(tname[0]), s) #,index_col='Date')
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError, IOError, EOFError) as e:
        print("Testing multiple exceptions. {}".format(e.args[-1]))
        pass
    if debug:
        print(stock.columns)
        print(stock)
    if 'Last' in stock:
        MakePlot(stock.Date, stock.Last, xname='Date',yname='Percentage '+tname[0],saveName=tname[0])
    try:
        distin = sc.execute('SELECT COUNT(DISTINCT Date) from %s' %tname[0]).fetchall()[0][0]
        allD = sc.execute('SELECT COUNT(Date) from  %s' %tname[0]).fetchall()[0][0]
        if abs(allD-distin)>0:
            print(distin,allD,tname[0])
    except sqlite3.OperationalError:
        print('Could not load!')
sc.close()


#https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey=demo
#https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey=demo
#https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey=demo
# https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey=demo
#ps://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=demo
#https://www.alphavantage.co/query?function=INFLATION&apikey=demo
#https://www.alphavantage.co/query?function=INFLATION_EXPECTATION&apikey=demo
#https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&apikey=demo
#https://www.alphavantage.co/query?function=RETAIL_SALES&apikey=demo
#https://www.alphavantage.co/query?function=DURABLES&apikey=demo
#https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=demo
#https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey=demo

#url = 'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=demo'
#r = requests.get(url)
#data = r.json()
#print(data)
