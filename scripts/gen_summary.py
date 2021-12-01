import requests,os
import pandas as pd
import matplotlib.pyplot as plt
import urllib3
draw=False
doPDFs=False
debug=False
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
ALPHA_ID = os.getenv('ALPHA_ID')

list_of_fundamentals = ['REAL_GDP&interval=quarterly',
                            'REAL_GDP&interval=annual','REAL_GDP_PER_CAPITA','TREASURY_YIELD&interval=monthly&maturity=10year','TREASURY_YIELD&interval=monthly&maturity=3month','TREASURY_YIELD&interval=monthly&maturity=30year',
                            'CPI&interval=monthly','INFLATION','INFLATION_EXPECTATION',
                            'CONSUMER_SENTIMENT','RETAIL_SALES','DURABLES','UNEMPLOYMENT','NONFARM_PAYROLL']
for f in list_of_fundamentals:
    if debug: print(f)
    url = 'https://www.alphavantage.co/query?function=%s&apikey=%s' %(f,ALPHA_ID) 
    r=None
    my_break = False
    while not my_break:
        try:
            r = requests.get(url)
            my_break=True
        except (ValueError,urllib3.exceptions.ProtocolError,ConnectionResetError,urllib3.exceptions.ProtocolError,ConnectionResetError) as e:
            print("Testing multiple exceptions. {}".format(e.args[-1]))
            continue
    data = r.json()
    if debug: print(data)
    if 'data' in data and len(data['data']):
        my_data = pd.DataFrame(data['data'])
        my_data['value'] = pd.to_numeric(my_data['value'],errors='coerce')
        my_data['date'] = pd.to_datetime(my_data['date'].astype(str), format='%Y-%m-%d')
        fout_name = f.replace('&','_').replace('=','_')+'_GLOBAL'
        MakePlot(my_data.date, my_data.value, xname='Date',yname='%s' %fout_name,saveName=fout_name)
    else:
        print('ERROR - %s' %url)
    #break

