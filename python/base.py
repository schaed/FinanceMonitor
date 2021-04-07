import math
import sys
import datetime
WAIT=False
style_path = '/Users/schae/testarea/CAFAna/HWWMVACode'
out_path = '/Users/schae/testarea/finances/yahoo-finance'
out_file_type = 'png'

def colorHTML(text, color='red',roundN=2):
    if roundN==2:
        return '<h7 style="color: %s;">%0.2f</h7>' %(color,text)        
    if roundN==4:
        return '<h7 style="color: %s;">%0.4f</h7>' %(color,text)
    return '<h7 style="color: %s;">%0.3f</h7>' %(color,text)

#-----------------------------------------------------
def makeHTMLIndex(outFileName,title, jetNames):
    with open(outFileName, 'w') as outFile:
        # write HTML header
        outFile.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title> Validation Plots </title>
        <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
        <script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        </head>
        <body>
        <div class="container">
        <h1> """+title+'-'+options.describe+""" </h1>
        <p>Last updated: {date}</p>
        </div>
        """.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        for jetName in jetNames:
            outFile.write("<a href=\""+jetName+"/"+jetName+".html\">"+jetName+"</a>\n")
        outFile.write("</body>\n")
        outFile.write("</html>")
        
#-----------------------------------------------------
def makeHTML(outFileName,title):

    plots = glob.glob('*.pdf')

    with open(outFileName, 'w') as outFile:
        # write HTML header
        outFile.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title> Validation Plots </title>
        <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
        <script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        </head>
        <body>
        <div class="container">
        <h1> Validation Plots </h1> 
        <p>Last updated: {date}</p> 
        </div>
        """.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

        plots = glob.glob('*.png')
        outFile.write("<h2> %s - %s </h2>" %(title,options.describe))
        outFile.write('<table style="width:100%">')
        outFile.write("<tr>\n")
        for i in range(0,len(plots)):
            offset = 2
            if i==0 or i%3==0:
                outFile.write("<tr>\n")
            outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"" + plots[i] + "\"><img src=\"" + plots[i] + "\" alt=\"" + plots[i] + "\" width=\"100%\"></a></td>\n")
            if (i>offset and (i-offset)%3==0) or i==len(plots): 
                outFile.write("</tr>\n")
        outFile.write("</tr>\n")
        outFile.write("</table>\n")

        outFile.write("</body>\n")
        outFile.write("</html>")
        
#-----------------------------------------------------
def makeHTMLTable(outFileName,title='Stock Performance', columns=[], entries=[]):
    
    with open(outFileName, 'w') as outFile:
        # write HTML header
        outFile.write("""
        <!DOCTYPE html>
        <html lang="en">
        <style>
        .tooltip {
          position: relative;
          display: inline-block;
          border-bottom: 1px dotted black;
        }
        
        .tooltip .tooltiptext {
          visibility: hidden;
          width: 120px;
          background-color: #555;
          color: #fff;
          text-align: center;
          border-radius: 6px;
          padding: 5px 0;
          position: absolute;
          z-index: 1;
          bottom: 125%;
          left: 50%;
          margin-left: -60px;
          opacity: 0;
          transition: opacity 0.3s;
        }
        
        .tooltip .tooltiptext::after {
          content: "";
          position: absolute;
          top: 100%;
          left: 50%;
          margin-left: -5px;
          border-width: 5px;
          border-style: solid;
          border-color: #555 transparent transparent transparent;
        }
        
        .tooltip:hover .tooltiptext {
          visibility: visible;
          opacity: 1;
        }
        </style>
        <head>
        <meta charset="utf-8">
        <title> """+title+""" </title>
        <script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet">
    <style type="text/css">
        h3 span {
            font-size: 22px;
        }
        h3 input.search-input {
            width: 300px;
            margin-left: auto;
            float: right
        }
        .mt32 {
            margin-top: 32px;
        }
    </style>
        </head>
        <body>
        <div class="container">
        <h1> """+title+""" </h1> 
        <p>Last updated: {date}</p>
        </div>
        """.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        line='    	<div class="container">\n'
        line+='    	<h3>\n'
        line+='    	    <span>Stock Info Search</span>\n'
        line+='    	    <input type="search" placeholder="Search..." class="form-control search-input" data-table="customers-list"/>\n'
        line+='    	</h3>\n'
        line+='    <table class="searchable sortable table table-striped mt32 customers-list">\n'
        line+=' <thead>\n'
        # generate the title
        line+='   <tr>\n'
        my_map ={'alpha':'An alpha of 1.0 means the fund has outperformed its benchmark index by 1. Correspondingly, an alpha of -1.0 would indicate an underperformance of 1. For investors, the higher the alpha the better.',
                     'beta':'A beta of 1.0 indicates that the investments price will move in lock-step with the market. A beta of less than 1.0 indicates that the investment will be less volatile than the market. Correspondingly, a beta of more than 1.0 indicates that the investments price will be more volatile than the market. For example, if a fund portfolios beta is 1.2, it is theoretically 20 more volatile than the market.',
                     'sharpe':'Sharpe Ratios above 1.00 are generally considered good, as this would suggest that the portfolio is offering excess returns relative to its volatility. Having said that, investors will often compare the Sharpe Ratio of a portfolio relative to its peers. Therefore, a portfolio with a Sharpe Ratio of 1.00 might be considered inadequate if the competitors in its peer group have an average Sharpe Ratio above 1.00.',
                     'rsquare':'R squared: 85 and 100 is like ETF, 70 is not like ETF, The R2 is a measure of how well the the returns of a stock is explained by the returns of the benchmark. If your investment goal is to track a particular benchmark, then you should chose stocks that show a high R2 with respect to the benchmark. The R2 value of 1 means that the benchmark completely explains the stock returns, while a value of 0 means that the benchmark does not explain the stock returns.',
                     'daily_return_stddev14':'Standard deviation measures the dispersion of data from its mean. Basically, the more spread out the data, the greater the difference is from the norm. In finance, standard deviation is applied to the annual rate of return of an investment to measure its volatility (risk). A volatile stock would have a high standard deviation. With mutual funds, the standard deviation tells us how much the return on a fund is deviating from the expected returns based on its historical performance.',
                     'rsi10':'oversold <20 and overbought>80',
                     'sma200':'Simple moving average of the last 200 days',
                     'sma100':'Simple moving average of the last 100 days',
                     'sma20':'Simple moving average of the last 20 days',
                     'CMF':'Chaikin Money Flows Value fluctuates between 1 and -1. CMF can be used as a way to further quantify changes in buying and selling pressure and can help to anticipate future changes and therefore trading opportunities',}
        for c in columns:
            #line+='     <th class="tooltip">'+c+'<span class="tooltiptext">Tooltip text</span></th>\n'
            if c in my_map:
                line+='     <th id="metricsHead" title="%s">%s</th>\n' %(my_map[c],c)
            else:
                line+='     <th>'+c+'</th>\n'
        line+='  </tr>\n'
        line+=' </thead>\n'
        # generate body
        line+=' <tbody>\n'

        for e in entries:
            line+='   <tr>\n'
            for i in e:
                try:
                    line+='    <td>%0.2f</td>\n' %(i)
                except:
                    line+='    <td>%s</td>\n' %(i)
            line+='   </tr>\n'
        # end table
        line+=' </tbody>\n'        
        outFile.write(line)
        outFile.write("</table>\n")
        outFile.write("</div>\n")

        outFile.write("</body>\n")
        outFile.write("""    <script>
        (function(document) {
            'use strict';

            var TableFilter = (function(myArray) {
                var search_input;

                function _onInputSearch(e) {
                    search_input = e.target;
                    var tables = document.getElementsByClassName(search_input.getAttribute('data-table'));
                    myArray.forEach.call(tables, function(table) {
                        myArray.forEach.call(table.tBodies, function(tbody) {
                            myArray.forEach.call(tbody.rows, function(row) {
                                var text_content = row.textContent.toLowerCase();
                                var search_val = search_input.value.toLowerCase();
                                row.style.display = text_content.indexOf(search_val) > -1 ? '' : 'none';
                            });
                        });
                    });
                }

                return {
                    init: function() {
                        var inputs = document.getElementsByClassName('search-input');
                        myArray.forEach.call(inputs, function(input) {
                            input.oninput = _onInputSearch;
                        });
                    }
                };
            })(Array.prototype);

            document.addEventListener('readystatechange', function() {
                if (document.readyState === 'complete') {
                    TableFilter.init();
                }
            });

        })(document);
    </script>""")
        outFile.write("</html>")
        
#------------------
def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    #print n,ss,data
    pvar = ss/float(n) # the population variance
    return pvar**0.5

#-----------------------------------------  
def GenerateToys(fsec, hbg, ntoys, m_rand, root):

    if ntoys<0:
        return
    
    params=[]
    initial_params=[]    
    # Load initial params
    for par in range(0,fsec.GetNumberFreeParameters()):
        params+=[fsec.GetParameter(par)]
        initial_params+=[fsec.GetParameter(par)]
    hbg_central=[]
    hbg_error=[]    
    hbg_bin_edge=[]
    for ibin in range(0,hbg.GetNbinsX()):
        hbg_central+=[hbg.GetBinContent(ibin)]
        hbg_error+=[0.0]
        hbg_bin_edge+=[(hbg.GetXaxis().GetBinUpEdge(ibin)-hbg.GetXaxis().GetBinLowEdge(ibin))/2.0+hbg.GetXaxis().GetBinLowEdge(ibin)]

    # Run
    for i in range(0,ntoys):
        if (i%1000)==0:
            print('Running Uncertainties for toy: %s' %i)
        
        # smear params and set
        for par in range(0,fsec.GetNumberFreeParameters()):
            params[par]=m_rand.Gaus(initial_params[par], fsec.GetParError(par))
            fsec.SetParameter(par, root.Double(params[par]))
        # Get difference with fit
        for ibin in range(0,hbg.GetNbinsX()):
            hbg_error[ibin]+=(hbg_central[ibin]-fsec.Eval(hbg_bin_edge[ibin]))**2

    # finish
    for ibin in range(0,hbg.GetNbinsX()):
        hbg.SetBinError(ibin,math.sqrt(hbg_error[ibin]/float(ntoys)))

    # Reset
    for par in range(0,fsec.GetNumberFreeParameters()):
        fsec.SetParameter(par,initial_params[par])
def DoRatio(ROOT):
    c1 = ROOT.TCanvas("c1","stocks",50,50,600,600);
    DORATIO=True
    padScaling=1.0
    ratioPadScaling=1.0
    pads=[]
    if DORATIO:
        #print 'doing ratio'
        # CAF setup
        ratioPadRatio  = 0.3;
        markerSize = 1;
        lineWidth = 2;
        markerStyle = 20;
        scale=1.05
        padScaling      = 0.75 / (1. - ratioPadRatio) *scale;
        ratioPadScaling = 0.75*(1. / ratioPadRatio) *scale;  
        ROOT.gStyle.SetPadTopMargin(0.065);
        ROOT.gStyle.SetPadRightMargin(0.05);
        ROOT.gStyle.SetPadBottomMargin(0.16);
        ROOT.gStyle.SetPadLeftMargin(0.16);
        ROOT.gStyle.SetTitleXOffset(1.0);
        pads=[]
        pads.append( ROOT.TPad('pad0','pad0', 0.0, ratioPadRatio, 1.0, 1.0) )
        pads.append( ROOT.TPad('pad1','pad1', 0.0, 0.0, 1.0, ratioPadRatio+0.012) )
        
        pads[0].SetTopMargin(padScaling * pads[0].GetTopMargin());
        pads[0].SetBottomMargin(.015);
        pads[0].SetTickx(True);
        pads[0].SetTicky(True);
        pads[1].SetTopMargin(.015);
        #pads[1].SetBottomMargin(ratioPadScaling *pads[1].GetBottomMargin());
        #print 'margin:',ratioPadScaling *pads[1].GetBottomMargin()
        pads[1].SetBottomMargin(ratioPadScaling *pads[1].GetBottomMargin()*0.93);    
        pads[1].SetGridy(True);
        pads[1].SetTickx(True);
        pads[1].SetTicky(True);
        pads[0].Draw()
        pads[1].Draw()
        pads[0].cd()
        #base.Format([h,htada],ROOT,True, padScaling,hist_name='')
    return c1,pads,padScaling,ratioPadScaling
#-----------------------------------------  
def SeperationPower(h1o, h2o):
    h1 = h1o.Clone()
    h2 = h2o.Clone()
    if h1.Integral()>0.0:
        h1.Scale(1.0/h1.Integral())
    if h2.Integral()>0.0:
        h2.Scale(1.0/h2.Integral())
    tot_sep = 0.0
    for i in range(1,h1.GetNbinsX()+1):
        #for j in range(1,h2.GetNbinsX()+1):
        bin_sum = h1.GetBinContent(i)+h2.GetBinContent(i)
        sep = 0.5*(h1.GetBinContent(i)-h2.GetBinContent(i))**2
        if bin_sum>0.0:
            sep /= bin_sum
        tot_sep += sep
            
    return tot_sep

#---------------
def Round(n,number_after_period=3):
    if n==0.0 or abs(n)<1.0e-10:
        return n
    log_n=0
    try:
        log_n = round(abs(math.log(abs(n),10)),0)
    except:
        print('ERROR with math log on %s' %n)
        log_n=0
    n_new = n*(10.0**(-log_n))
    n_new2 = round(n_new,number_after_period)
    
    return n_new2*(10.0**(log_n))

#---------------
def RoundScientific(n,numbers=3):
    if n==0.0 or abs(n)<1.0e-10:
        return n
    log_n=0
    try:
        log_n = round(abs(math.log(n,10)),0)
    except:
        print('ERROR with math log on %s' %n)
        log_n=0
    n_new = n*(10.0**(-log_n))
    if n_new<1.0:
        n_new*=10.0
        log_n-=1    
    n_new2 = round(n_new,numbers)
    return '%0.2fx10^{%0.0f}' %(n_new2,log_n)
    #return n_new2*(10.0**(log_n))

#---------------
def RoundStr(n,number_after_period=3):
    n = Round(n,number_after_period=3)
    s = '%s' %n
    if (len(s)-s.find('\.'))>(number_after_period+1):
        s = '%0.3f' %n

    return s

#-----------------------------------------  
def Style(root):
    sys.stdout.flush()
    if not hasattr(root,'SetAtlasStyle'):
        root.gROOT.LoadMacro(style_path+'/atlasstyle-00-03-05/AtlasStyle.C')
        root.gROOT.LoadMacro(style_path+'/atlasstyle-00-03-05/AtlasUtils.C')
        root.SetAtlasStyle()

#-----------------------------------------
def Format(mcs, ROOT, isData=False, ratioPadScaling=1.0, hist_name='',addOverflow=True):

    for m in mcs:

        if m==None or not m:
            continue
        
        if ratioPadScaling!=1.0:
            m.SetMarkerSize(0.4)
            bx = m.GetXaxis();
            by = m.GetYaxis();
            bx.SetTitleSize(ROOT.gStyle.GetTitleSize("x") * ratioPadScaling);
            bx.SetLabelSize(ROOT.gStyle.GetLabelSize("x") * ratioPadScaling);
            by.SetTitleSize(ROOT.gStyle.GetTitleSize("y") * ratioPadScaling);
            by.SetTitleOffset(ROOT.gStyle.GetTitleOffset("y") / ratioPadScaling  );

            if hist_name.count('_recoilrms'):
                bx.SetTitleOffset(ROOT.gStyle.GetTitleOffset("x") *0.96  );
            else:
                bx.SetTitleOffset(ROOT.gStyle.GetTitleOffset("x") *1.19  );
            by.SetLabelSize(ROOT.gStyle.GetLabelSize("y") * ratioPadScaling);
            bx.SetLabelColor(1);
            bx.SetTickLength(bx.GetTickLength() * ratioPadScaling);

        if addOverflow: # stack underflow and overflow
            try:
                a0=m.GetBinContent(0)
                e0=m.GetBinError(0)
                e1=m.GetBinError(0)
                m.SetBinContent(1,a0+m.GetBinContent(1))
                m.SetBinError(1, math.sqrt(e0**2+e1**2))
                m.SetBinContent(0,0.0)
                m.SetBinError(0,0.0)
                last_bin = m.GetNbinsX()
                m.SetBinContent(last_bin,m.GetBinContent(last_bin)+m.GetBinContent(1+last_bin))
                m.SetBinError(last_bin,math.sqrt(m.GetBinError(last_bin)**2+m.GetBinError(1+last_bin)**2))
                m.SetBinContent(last_bin+1,0.0)
                m.SetBinError(last_bin+1,0.0)

            except AttributeError:
                pass

            
#-------------------------------------------------------------------------                                                                                                                                    
def getATLASLabels(pad, x, y, ROOT, text='', padScaling=1.0, lumi=1.4):

    l = ROOT.TLatex(x, y, 'ATLAS')
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextSize(0.055*padScaling)
    l.SetTextAlign(11)
    l.SetTextColor(ROOT.kBlack)
    l.Draw()

    delx = 0.05*pad.GetWh()/(pad.GetWw())
    labs = [l]

    if True:
        p = ROOT.TLatex(x+0.15, y, 'Internal')
        p.SetNDC()
        p.SetTextFont(42)
        p.SetTextSize(0.055*padScaling)
        p.SetTextAlign(11)
        p.SetTextColor(ROOT.kBlack)
        p.Draw()
        labs += [p]

        a = ROOT.TLatex(x, y-0.045*padScaling, '#sqrt{s} = 13 TeV Simulation')        
        if text.count('Data'):
            a = ROOT.TLatex(x, y-0.045*padScaling, '#sqrt{s} = 13 TeV, L = %.1f fb^{-1}' %(lumi))
        a.SetNDC()
        a.SetTextFont(42)
        a.SetTextSize(0.04*padScaling)
        a.SetTextAlign(12)
        a.SetTextColor(ROOT.kBlack)
        a.Draw()
        labs += [a]

    if text!='':
        c = ROOT.TLatex(x, y-0.1*padScaling, text)
        c.SetNDC()
        c.SetTextFont(42)
        c.SetTextSize(0.04*padScaling)
        c.SetTextAlign(12)
        c.SetTextColor(ROOT.kBlack)
        c.Draw()
        labs += [c]

    return labs


def getLabels(pad, x, y, ROOT, text='', padScaling=1.0):

    labs=[]
    c = ROOT.TLatex(x, y-0.13*padScaling, text)
    c.SetNDC()
    c.SetTextFont(42)
    c.SetTextSize(0.04*padScaling)
    c.SetTextAlign(12)
    c.SetTextColor(ROOT.kBlack)
    c.Draw()
    labs += [c]
    return labs

#---------------------------------------------------------------------
# Make logger object
#
def getLog(name, level = 'INFO', debug=False):

    import logging
    import sys
    
    f = logging.Formatter("Py:%(name)s: %(levelname)s - %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(f)
    
    log = logging.getLogger(name)
    log.addHandler(h)

    if debug:
        log.setLevel(logging.DEBUG)
    else:
        if level == 'DEBUG':   log.setLevel(logging.DEBUG)
        if level == 'INFO':    log.setLevel(logging.INFO)
        if level == 'WARNING': log.setLevel(logging.WARNING)    
        if level == 'ERROR':   log.setLevel(logging.ERROR)

    return log

#-------------------------------------------------------------------------
# Common command line option parser
#
def getParser():
    
    from optparse import OptionParser
    
    p = OptionParser(usage='usage: <path:ROOT file directory>', version='0.1')

    #
    # Options for plotEvent.py
    #
    p.add_option('-n','--nevent',  type='int',    dest='nevent',         default=0,         help='number of events')
    p.add_option('--ntoys',  type='int',    dest='ntoys',         default=-1,         help='number of toys for systematics varying fit uncertainties')    
    p.add_option('-i','--input',  type='string',    dest='input',         default=None,         help='Files to fit in a comma separated list')
    #p.add_option('-s','--sample',  type='string',    dest='data',         default='data',         help='sample')        
    p.add_option('--hname',     type='string',    dest='hname',         default=None,         help='Histogram name')
    p.add_option('--fit-dir',   type='string',    dest='fit_dir',      default='fitfuncs',         help='Fit function directory')
    p.add_option('--fit-name',  type='string',    dest='fit_name',    default='FourParamFit',         help='Fit Function name from fitfunc.py')
    p.add_option('--fit-opt',   type='string',    dest='fit_opt',    default='M',         help='Fit Function name from fitfunc.py')
    p.add_option('--add-lumi-dir',   type='string',    dest='add_lumi_dir',    default=None,         help='Add lumi dir')        

    p.add_option('--xmin',      type='float',    dest='xmin',         default=1100.0,         help='Fit range min')    
    p.add_option('--xmax',      type='float',    dest='xmax',         default=6100.0,         help='Fit range max')    
    p.add_option('--xmin-fit',  type='float',    dest='xmin_fit',         default=3570.0,         help='Fit range to exclude min')    
    p.add_option('--xmax-fit',  type='float',    dest='xmax_fit',         default=3990.0,         help='Fit range to exclude max')    
    p.add_option('--lumi',      type='float',    dest='lumi',         default=1.6,         help='Luminosity')
    p.add_option('--scale-by-lumi',      type='float',    dest='scale_by_lumi',         default=None,         help='Scale data and bkg by lumi')
    p.add_option('--scale-by-toys',      type='int',    dest='scale_by_toys',         default=1,         help='Scale data and bkg by lumi')    
    p.add_option('--scale-by-lumi-sig',      type='float',    dest='scale_by_lumi_sig',         default=None,         help='Scale data and sig by lumi')            

    p.add_option('--wait',        action='store_true', default=False,    dest='wait',     help='wait on plots')
    p.add_option('--overlay-fits',        action='store_true', default=False,    dest='overlay_fits',     help='overlay-fits')    
    p.add_option('--tada',        action='store_true', default=False,    dest='tada',     help='tada')    
    p.add_option('--fit-mc',      action='store_true', default=False,    dest='fit_mc',     help='Fit MC shape')
    p.add_option('--fit-mc-only', action='store_true', default=False,    dest='fit_mc_only',     help='Fit MC shape only')        
    p.add_option('--do-ratio',    action='store_false',default=True,     dest='do_ratio', help='do ratio plot')    
    
    return p

#-------------------------------------------------------------------------
# stock list
#
stock_lista = [
                  ['SPY',200.0,805.0,'NYSE','spy'], # spy
                  ['QQQ',200.0,805.0,'NYSE','qqq'], # nasdaq
                  ['GOOGL',640.0,805.0,'NASDAQ','google'], # google                  
        ['AMZN',450.0,700.0,'NASDAQ','amazon'], # amazon
        ['AAPL',86.0,110.0,'NASDAQ','apple'], # apple
        ['MAT',25.0,40.0,'NYSE','matel'], # matel
        ['FB',93.0,130.0,'NASDAQ','facebook'],
        ['X',20.0,55.0,'NYSE','PA steel'],  # steel industry
        ['RIOT',20.0,55.0,'NYSE','block chaing'],  #
        ['TLRY',20.0,55.0,'NYSE','pot stock'],  #        
        ['XME',20.0,55.0,'NYSEARCA','s&p metals and miners'],  # 0.8% dividend
        ['CLF',2.0,55.0,'NYSE','OH - iron ore'],  # iron ore company
        ['FLR',2.0,155.0,'NYSE','construction TX'],  # construction. texas 1.5%
        ['FLIR',2.0,155.0,'NYSE','heat camera'],  # 
        ['F',2.0,155.0,'NYSE','Ford'],  # 
        ['V',2.0,155.0,'NYSE','VISA'],  # 
        ['CSCO',2.0,155.0,'NYSE','cisco'],  # 
        ['PLTR',2.0,155.0,'NYSE','palantir'],  # 
        ['GLDD',2.0,155.0,'NASDAQ','dregding & land reclaim'],  # Great lakes dredge and dock
        ['NUE',2.0,155.0,'NYSE','nucor 2.5% mini-steel maker'],  # nucor 2.5% mini-steel maker
        ['GVA',2.0,155.0,'NYSE',' granite. civil engineering firm.'],  # granite. civil engineering firm. california. 0.95%
        #['SUM',2.0,155.0,'NYSE','summit materials (denver)'],  # summit materials (denver)
        ['SCCO',20.0,55.0,'NYSE','copper company'],  # copper company 0.5%
        ['SPR',20.0,105.0,'NYSE','spirit airlines'],  # spirit airlines 0.7%
        #['SFLY',20.0,105.0,'NASDAQ','shutterfly'],  # shutterfly                  
                  ]
stock_list = [
        # Check stocks
        ['GOOGL',640.0,805.0,'NASDAQ','google'], # google
        ['AMZN',450.0,700.0,'NASDAQ','amazon'], # amazon
        ['QQQ',200.0,805.0,'NYSE','qqq'], # nasdaq
        ['SPY',200.0,805.0,'NYSE','spy'], # spy
        ['AAPL',86.0,110.0,'NASDAQ','apple'], # apple
        ['PLTR',2.0,155.0,'NYSE','palantir'],  # 
        ['MAT',25.0,40.0,'NYSE','matel'], # matel
        ['CLPS',25.0,40.0,'NYSE','CLPS'], # CLPS
        ['JFU',25.0,40.0,'NYSE','JFU'], # JFU
        ['HAS',25.0,120.0,'NYSE','hasbro'], # hasbro
        ['TLRY',25.0,120.0,'NYSE','Tilray'], # Tilray
        ['VTWO',25.0,120.0,'NYSE','Top 2000'], # Top 2000
        ['RIOT',25.0,120.0,'NYSE','RIOT'], # RIOT        
        ['MJ',25.0,120.0,'NYSE','Marjo ETF'], # MJ        
        ['FB',93.0,130.0,'NASDAQ','facebook'],
        ['X',20.0,55.0,'NYSE','PA steel'],  # steel industry
        ['XME',20.0,55.0,'NYSEARCA','s&p metals and miners'],  # 0.8% dividend
        ['CLF',2.0,55.0,'NYSE','OH - iron ore'],  # iron ore company
        ['MT',2.0,55.0,'NYSE','mining iron'],  # steel production
        ['FLR',2.0,155.0,'NYSE','construction TX'],  # construction. texas 1.5%
        ['GLDD',2.0,155.0,'NASDAQ','dregding & land reclaim'],  # Great lakes dredge and dock
        ['NUE',2.0,155.0,'NYSE','nucor 2.5% mini-steel maker'],  # nucor 2.5% mini-steel maker
        ['GVA',2.0,155.0,'NYSE',' granite. civil engineering firm.'],  # granite. civil engineering firm. california. 0.95%
        #['SUM',2.0,155.0,'NYSE','summit materials (denver)'],  # summit materials (denver)
        ['SCCO',20.0,55.0,'NYSE','copper company'],  # copper company 0.5%
        ['SPR',20.0,105.0,'NYSE','spirit airlines'],  # spirit airlines 0.7%
        #['SFLY',20.0,105.0,'NASDAQ','shutterfly'],  # shutterfly
        ['NLSN',30.0,68.0,'NYSE','Nielsen'],  # 3% div. Nielsen
        ['PG',30.0,68.0,'NYSE','P&G'],  # 3% div. P&G
        ['UN',30.0,68.0,'NYSE','unilever'],  # 3% div. unilever
        ['UCTT',3.0,68.0,'NASDAQ','ultra-clean holdings'],  # ultra clean holdings
        ['DE',30.0,608.0,'NYSE','john deere'],  # 3% div. john deere
        #['MON',30.0,608.0,'NYSE','Monsanto'],  # 2% div. Monsanto
        ['SNA',30.0,558.0,'NYSE','snap on'],  # snap on. 1.6%
        ['MPC',30.0,98.0,'NYSE','marathon gas refinery'],  # marathon gas refinery
        ['OXY',30.0,98.0,'NYSE','occidental-fracing'],  # occidental petrol. 4.5%
        ['CRZO',20.0,98.0,'NASDAQ','carrizo oil. drilling and wells'],  # carrizo oil & gass
        ['CCJ',20.0,98.0,'NASDAQ','sells uranium'],  # CCJ uranium    
        ['SGG',5.0,98.0,'NASDAQ','sugar ETF'],  # sugar SGG
        ['EOG',30.0,198.0,'NYSE','eog-fracing'],  # fracing faster growing 0.7%
        ['PXD',30.0,298.0,'NYSE','Pioneer-fracing'],  # fracing faster growing 0.07%    
        ['WES',30.0,98.0,'NYSE','western gas'],  # western gas. 5%    
        ['WNR',25.0,80.0,'NYSE','western refinery'],  # western refinery 4.% dividend.
        ['CHK',4.0,7.0,'NYSE','cheseapeak oil'],  # cheseapeak
        ['FSLR',10.0,500.0, 'NASDAQ','first solar'], #first solar, arizona based
    #['SUNEQ',10.0,500.0, 'NASDAQ'], #first solar, arizona based
        ['SPWR',10.0,500.0, 'NASDAQ','sun power'], # sun power, san jose based
        ['SO',10.0,500.0, 'NYSE','southern co'], # southern co, 4.7%
        ['TSL',10.0,500.0, 'NYSE','trina solar limited, chinese'], # trina solar limited, chinese
        ['EIX',10.0,500.0, 'NYSE','edison solar'], # edison international, 2.8% solar
        ['NEE',10.0,500.0, 'NYSE','nextera broad energy'], # nextera energy, 2.7%, florida
        ['PCG',10.0,500.0, 'NYSE','PG&E energy CA'], # PG&E, 3%, san fransico
        ['CPRI',45.0,60.0,'NYSE','KORS'], # cosmetics
        ['NGL',5.0,15.0,'NYSE','pipeline company'], # pipeline company
        ['ETP-E',5.0,55.0,'NYSE','pipeline company'], # pipeline company. 11%
        #['ETE',5.0,55.0,'NYSE','pipeline company'], # pipeline company. 5.9%    
        ['CVX',78.0,100.0,'NYSE','chevron'], # chevron
        ['UAA',35.0,50.0,'NYSE','under armour'], # under armour
        ['KR',35.0,50.0,'NYSE','kroger'], # kroger. 1%
        ['SKT',25.0,50.0,'NYSE','tanger outlet'], # tanger, 3.5% dividend  
        ['TGT',65.0,85.0,'NYSE','target'], # target. 3%        
        ['CVS',80.0,120.0,'NYSE','CVS'], # CVS 1.6%
        ['WBA',80.0,120.0,'NYSE','walgreens'], # walgreens 3.6%
        ['COST',80.0,120.0,'NYSE','costco'], # walgreens 0.9%
        #['TFM',25.0,35.0,'NASDAQ','fresh market'], # fresh market
        ['SFM',25.0,35.0,'NASDAQ','sprouts farms'], # sprouts farms
        #['WFM',25.0,35.0,'NASDAQ','whole foods'], # whole foods 1.7%
        ['CMG',400.0,600.0,'NYSE','chipotle'], # chipotle
        ['JACK',60.0,80.0,'NASDAQ','JACK in the box'], # JACK in the box
        ['WEN',9.0,15.0,'NASDAQ','wendys'], # wendys
        ['PZZA',50.0,65.0,'NASDAQ','papa johns'], # papa johns
        ['MCD',100.0,150.0,'NYSE','mc donalds'], # mc donalds - 3%
        ['DIN',60.0,120.0,'NYSE','IHOP'], # IHOP 3.7%
        ['DENN',7.0,15.0,'NASDAQ','dennys'], # dennys - None
        ['F',10.0,15.0,'NYSE','ford'], # ford
        ['GM',25.0,40.0,'NYSE','GM'], # GM
        ['TM',25.0,200.0,'NYSE','toyota'], # toyota 3.3%
        ['HMC',25.0,200.0,'NYSE','honda'], # honda 2%
        ['THO',10.0,150.0,'NYSE','thor. sports utility vehicle'], # thor. sports utility vehicles
        ['VZ',45.0,55.0,'NYSE','verizon'], # verizon
        ['AMT',45.0,155.0,'NYSE','connection tower company'], # connection tower company. 2.2% dividend
        ['M',35.0,55.0,'NYSE','macys'], # macy's
        ['TU',1.0,55.0,'NASDAQ','tuesday morning corp'], # tuesday morning corp
        ['SXI',1.0,155.0,'NYSE','standex'], # standex
        ['MMM',132.0,170.0,'NYSE','3M'], # 3M
        ['TSO',50.0,105.0,'NYSE','Tesoro'], # Tesoro
    #['NTI',20.0,30.0], # northern tier refinery. pays 15 % dividend
        ['INTC',25.0,34.0,'NASDAQ','intel'], # intel 3.55% dividend
        ['NVDA',25.0,234.0,'NASDAQ','nvidia-graphics processing chips'], # nvidia 0.55% dividend    
        ['BCS',5.0,15.0,'NYSE','barclays'], # barclays
        ['CS',5.0,15.0,'NYSE','credit suisse'], # credit suisse banking stock. 6.7% dividend
        ['UBS',8.0,20.0,'NYSE','UBS'], # ubs. 6.4% dividend
        ['DB',8.0,20.0,'NYSE','deuchee bank'], # deuchee bank.
        ['EBAY',20.0,30.0,'NASDAQ','EBAY'], # ebay
    ['BAC',10.0,50.0,'NYSE','bank of america'], # bank of america. 1.2%
    #['MS',10.0,80.0,'NYSE'], # morgan stanley 1.7% 
        ['UNH',100.0,130.0,'NYSE','united health care'], # united health care
        ['CI',120.0,180.0,'NYSE','health care. cigna'], # health care. cigna
        ['PFE',25.0,38.0,'NYSE','pfizer'], # pfizer 4% dividend
        #['AET',75.0,120.0,'NYSE','aetna'], # aetna 1% dividend
        #['TDOC',75.0,120.0,'NYSE','teladoc'], # online doctor
        ['HUM',145.0,190.0,'NYSE','humara'], # humara 1% dividend
        ['TFX',120.0,160.0,'NYSE','teleflex medical device, Wayne, PA'], # teleflex 1% dividend. medical devices. wayne, PA
        ['FMS',10.0,160.0,'NYSE','Fresenius Medical supply, Germany'], # Fresenius 1% dividend. medical supply
        ['LMAT',10.0,16.0,'NASDAQ','le maitre'], # le maitre 1% dividend. medical devices.
        ['MSEX',23.0,35.0,'NASDAQ','NJ water company'], # NJ water company. 2.7% dividend
        ['WTR',30.0,40.0,'NYSE','PA water company'], # PA water company
        ['AWK',60.0,75.0,'NYSE','canada water company'], # canada water company
        ['AWR',20.0,53.0,'NYSE','american states water company'], # american states water company
        ['PNR',20.0,53.0,'NYSE','pentair. partial water company'], # pentair. partial water company that may grow
        ['DUK',70.0,100.0,'NYSE','DUKE energy.'], # DUKE energy. good electric stock. 3.8% dividend
        ['PPL',30.0,80.0,'NYSE','good electric stock'], # PPL comp. good electric stock. 4% dividend
        ['XYL',40.0,75.0,'NYSE','water tech company'], # water tech company
        ['DPS',85.0,105.0,'NYSE','dr pepper'], # dr pepper
        ['CINF',52.0,70.0,'NASDAQ','insurance. cincy'], # insurance. cincy. 3% dividend
        ['GILD',75.0,110.0,'NASDAQ','gilead biotech'], # gilead biotech
        ['AMGN',130.0,170.0,'NASDAQ','biotech. california'], # biotech. california. 2.7% dividend
        ['BIIB',200.0,300.0,'NASDAQ','Biogen biotech'], # Biogen biotech. california. 0.0% dividend
        ['SLP',7.0,15.0,'NASDAQ','Simulations Plus'], # Simulations Plus. 1.8% dividend. biomedical
        ['ADR',50.0,85.0,'NYSE','novartis'], # novartis    
        ['GVP',2.0,3.0,'NYSEMKT','GSE nuclear, oil simulations'], # GSE nuclear, oil simulations company
        ['TAP',80.0,100.0,'NYSE','molson beer'], # molson beer. 1.8% dividend
        ['RTN',115.0,160.0,'NYSE','ratheon'], # ratheon. defense. 2.1% dividend
        ['CXW',15.0,300.0,'NYSE','corecivics. jailing'], # corecivics. jailing. 5.8% dividend
        ['GEO',15.0,300.0,'NYSE','geo group. jailing service florid'], # geo group. jailing service florida. 6.4%   
        ['GD',115.0,260.0,'NYSE','general dynamics'], # general dynamics. 1.6%
        ['GE',15.0,300.0,'NYSE','general electric'], # general dynamics corp. 3.2%
        ['SID',1.0,5.0,'NYSE','steel in brazil'], # steel in brazil
        ['VLO',45.0,70.0,'NYSE','valero'], # oil refinery 3.9% dividend
        ['ABBV',50.0,70.0,'NYSE','pharma'], # pharma 4.0% dividend
        ['GLBS',5.0,70.0,'NASDAQ','globus maritime'], # globus maritime
        ['WDC',35.0,80.0,'NYSE','western digital'], # western digital 4.2% dividend
        ['STX',30.0,50.0,'NYSE','seagate'], # seagate 7% dividend
        ['BLK',200.0,500.0,'NYSE','black rock'], # black rock 2.9% dividend.
        ['CLGX',20.0,500.0,'NYSE','Corelogic. financial services'], # CoreLogic
        ['ADC',20.0,50.0,'NYSE','real estate'], # real estate 4.9% dividend.
        ['NTRI',10.0,30.0,'NASDAQ','nutrisystem'], # nutrisystem 4.0% dividend.
        ['MET',30.0,60.0,'NYSE','insurance'], # insurance 3.8% dividend. 
        ['WY',20.0,35.0,'NYSE','timber'], # real estate 5.% dividend. 
        ['RYN',20.0,35.0,'NYSE','florida timber'], # timber 3% dividend
        ['GLD',108.0,125.,'NYSE','gold'], # gold
        ['SLV',10.0,125.,'NYSE','silver ETF'], # silver
        ['USLV',10.0,125.,'NYSE','silver 3x ETF'], # silver
        ['SIL',10.0,125.,'NYSE','silver miners ETF'], # silver
    #['SHNY',10.0,125.,'NYSE','silver miners 2X ETF'], # silver
        ['USO',5.0,25.,'NYSE','crude oil'], # crude oil    
        ['GDX',11.0,125.,'NYSEARCA','gold miners'], # gold miners
        ['NUGT',1.0,125.0,'NYSEMKT','goldx5'], # gold    
        ['DIA',120.0,200.0,'NYSE','Dow jones'], # Dow jones 
        ['NDAQ',40.0,70.0,'NASDAQ','nasdaq trader'], # nasdaq trader. 1.7% dividend
        ['TSN',40.0,70.0,'NYSE','tyson foods'], # tyson foods. 1.% dividend
        ['ANDE',40.0,70.0,'NASDAQ','andersons fertilzer comp, OH'], # andersons fertilzer comp. 1.7% dividend    
        ['GSK',35.0,70.0,'NYSE','Glaxo-Smith Kline pharma'], # pharma. 6.% dividend 
        ['BMY',55.0,70.0,'NYSE','Bristol-Myers Squibb.'], # Bristol-Myers Squibb. 2.75% dividend
        ['LOW',55.0,100.0,'NYSE','LOWES'], # LOWES 2.% dividend
        ['HD',85.0,180.0,'NYSE','Home depot'], # Home depot 2.% dividend     
        ['CRM',50.0,75.0,'NYSE','salesforce'], # salesforce. cloud platform service. 0.% dividend. nielsen is using them       
        ['ADP',75.0,100.0,'NYSE','automatic data processing'], # automatic data processing. cloud platform service. 2.% dividend. 
        ['INFY',15.0,25.0,'NYSE','infosys. IT/software company'], # infosys. IT/software company. 2.% dividend          
        #['TCS',2000.0,2800.0,'TCS. IT/software company'], # TCS. IT/software company. 1.7% dividend          
        ['MCK',130.0,200.0,'NYSE','mckessen. health care robotics and machine'], # mckessen. health care robotics and machine dosing. 0.7% dividend
        ['BHP',15.0,40.0,'NYSE','BHP billington. mining company'], # BHP billington. mining company with 11% dividend. steve's pick. not so sure about this one
        ['BP',20.0,45.0,'NYSE','british patroleum'], # british patroleum.  8.2% dividends
        #['NEE',100.0,140.0,'florida electrical'], # florida electrical company.  2.7% dividends        
        ['ABX',8.0,20.0,'NYSE','Barrick Gold mining compan'], #Barrick Gold mining company 0.6% dividend
        ['SLW',10.0,20.0,'NYSE','Silver Wheaton Corp'], # Silver Wheaton Corp 1.6% dividend
        ['EXK',1.0,4.0,'NYSE','Silver mine'], # Silver mine 22% dividend
        #['HCHDF',0.2,2.0], # Silver mine 22% dividend
        ['GG',10.0,20.0,'NYSE','Goldcorp mining'], # Goldcorp mining company 1.6% dividend
        ['NEM',16.0,35.0,'NYSE','Newmont Mining gold mining'], # Newmont Mining gold mining company 0.4% dividend
        ['AUY',1.3,3.4,'NYSE','Yamana Gold mining'], # Yamana Gold mining company 2.5% dividend Canada
        ['NOA',1.3,4.0,'NYSE','Mining US.'], # Mining US. 2% dividend.
        ['PTON',1.3,400.0,'NYSE','peloton'], # peloton stock
        ['HMY',2.3,5.0,'NYSE','Harmony gold Mining US'], # Harmony gold Mining US.
        ['GFI',3.3,6.0,'NYSE','Gold fields unlimited. south african gold'], # Gold fields unlimited. south african gold
        ['EGO',3.3,6.0,'NYSE','eldarado gold'], # eldarado gold.
        ['BTG',1.3,4.0,'NYSEMKT','b2gold'], # b2gold
        ['VALE',3.3,6.0,'NYSE','mineral miner in brazil'], # mineral miner in brazil 2.dividend
        ['KGC',1.3,3.8,'NYSE','KinCross Gold mining'], # KinCross Gold mining company 0.0% dividend Canada
        ['CSCO',15.0,35.0,'NASDAQ','cisco'], # cisco. 3.7%
        ['NOVN',15.0,35.0,'NASDAQ','novn'], #
        ['CLOV',15.0,35.0,'NASDAQ','CLOV'], #  pharma       
        ['KMI',10.0,40.0,'NYSE','kinder morgan-oil gas pipelines'], # kinder morgan. Berkshire hathaway is investing in them. 2.9%

        ['GNOW',0.1,2.0,'NASDAQ','urgent care nano cap'], # urgent care nano cap. check carefully
        ['ENSG',10.0,25.0,'NASDAQ','urgent care small cap'], # urgent care small cap. check carefully
        ['ADPT',45.0,70.0,'NYSE','urgent care small cap'], # urgent care small cap. check carefully. no dividend
        ['EVHC',18.0,33.0,'NYSE','urgent care mid cap'], # urgent care mid cap. check carefully. no dividend
        ['LPNT',60.0,80.0,'NASDAQ','urgent care mid cap'], # urgent care mid cap. check carefully. no dividend
        ['THC',23.0,30.0,'NYSE','urgent care/intensive care mid cap'], # urgent care/intensive care mid cap. check carefully. no dividend

        ['SHAK',33.0,60.0,'NYSE','shake shack.'], # shake shack.
        ['UAL',45.0,90.0,'NYSE','united airlines'], # united airlines

        ['VOO',100.0,200.0,'NYSE','vanguard MUTF'], # vanguard MUTF
        ['VFINX',100.0,200.0,'MUTF','vanguard MUTF'], # vanguard MUTF
        ['VFIAX',100.0,200.0,'VFIAX','vanguard MUTF'], # vanguard MUTF
        ['VPU',75.0,125.0,'NYSE','vanguard utilities'], # vanguard utilities, 3.3% dividend
        ['RYU',50.0,100.0,'NYSE','equal weight utilities'], # equal weight utilities,
        ['VBK',50.0,170.0,'NYSE','vanguard small cap growth'], # vanguard small cap growth
        ['VYM',60.0,90.0,'NYSE','vanguard large cap'], # vanguard large cap mutual fund 3.1% dividend
        ['IVE',70.0,130.0,'NYSE','ishare mutual fund'], # ishare mutual fund 
        ['TSLA',200.0,300.0,'NASDAQ','tesla'], # tesla mototrs
        ['QS',200.0,300.0,'NASDAQ','QS'], #
        ['BBBY',40.0,70.0,'NASDAQ','bed bath and beyond'], # bed bath and beyond
        ['VA',30.0,70.0,'NASDAQ','virgin atlantic'], # virgin atlantic
        ['TWTR',10.0,20.0,'NYSE','twitter'], # twitter
        ['S',2.0,5.0,'NYSE','sprint'], # sprint
        ['TMUS',2.0,5.0,'NASDAQ','t-mobile'], # t-mobile
        ['HOG',40.0,55.0,'NYSE','harley davidson'], # 3% dividend harley davidson
        ['PIR',5.0,10.0,'NYSE','pier 1'], # 3% dividend pier 1 imports
        ['DDD',15.0,25.0,'NYSE','3D printing'], # 3D printing manufacturer
        ['XONE',10.0,15.0,'NASDAQ','3D printing exone'], # 3D printing manufacturer exone
        ['SSYS',20.0,40.0,'NASDAQ','3D printing'], # 3D printing manufacturer exone
        ['AMAT',13.0,27.0,'NASDAQ','chip gear manufacturer'], # chip gear manufacturer
        ['ROKU',10.0,19.0,'NASDAQ','ROKU'], #
        ['DKNG',10.0,19.0,'NASDAQ','draft kings'], #        
        ['GPRO',10.0,19.0,'NASDAQ','go pro'], # go pro stock        
        ['QCOM',40.0,65.0,'NASDAQ','qualcomm'], # qualcomm - starting in drone market. 4% dividend
        ['IXYS',10.0,15.0,'NASDAQ','parts manufacturer for drones'], # parts manufacturer for drones
        ['INVN',4.0,10.0,'NASDAQ','motion control US'], # parts manufacturer for drones. motion control
        ['STM',4.0,8.0,'NYSE','Geneva Semiconductor'], # parts manufacturer for drones. motion control. geneva based. won apple smart watch bid. 7.4% dividend
        ['MXL',16.0,30.0,'NYSE','maxlinear semiconductor'], # semiconductor manu.
        ['NBIX',40.0,60.0,'NASDAQ','bio pharma company'], # random bio pharma company. 0- dividend
        ['WB',20.0,60.0,'NASDAQ','Chinese social media'], # random company
        ['NXPI',70.0,100.0,'NASDAQ','dutch semi-conductor'], # semi-conductor manufacturer
        ['TXN',50.0,70.0,'NASDAQ','texas instraments. semi-conductor'], # texas instraments. semi-conductor manufacturer
        ['INFN',10.0,20.0,'NASDAQ','infera semi-conductor'], # infera semi-conductor manufacturer.
        ['LMT',150.0,300.0,'NYSE','lockheed martin'], # lockheed martin. 2.92
        ['BA',100.0,150.0,'NYSE','boeing'], # boeing
        ['NOC',170.0,250.0,'NYSE','northrop gruman'], # northrop gruman. 2.92
        ['GBSN',3.0,8.0,'NASDAQ','genetics testing'], # genetics testing company
        ['AMAG',20.0,50.0,'NASDAQ','pharma in iron deficiency'], # pharma in iron deficiency
        ['MOH',60.0,80.0,'NYSE','Molina health'], # Molina health. zach's #1
        ['CRL',70.0,100.0,'NYSE','Charles river health'], # Charles river health. zach's #2
        ['AIRM',30.0,50.0,'NASDAQ','air drop pharma'], # air drop pharma. zach's #2
        ['PRXL',60.0,71.0,'NASDAQ','paralex medical supplies'], # paralex medical supplies        
        ['FPRX',40.0,60.0,'NASDAQ','therapuetics'], # therapuetics - rated a buy.        
        ['EBS',30.0,50.0,'NYSE','emergent bio solutions'], # emergent bio solutions. high zacks rating
        ['FCSC',1.0,3.0,'NASDAQ','fibrocell'], # fibrocell. random pharma
        ['GENE',2.0,3.0,'NASDAQ','genetics testing'], # genetics testing company
        ['OPK',8.0,13.0,'NYSE','genetics testing'], # genetics testing company. is subsidiary
         ['RGLS',6.0,10.0,'NASDAQ','bio pharma'], # bio pharma
         ['DGX',50.0,90.0,'NYSE','pharma testing company'], # pharma testing company
         ['ORPN',2.0,5.0,'NASDAQ','bio pharma'], # bio pharma
         ['VIVO',15.0,25.0,'NASDAQ','Meridian malaria'], # malaria indicator stock. Meridian
         ['XON',30.0,50.0,'NYSE','Intrexon Zika?'], # zika indicator stock. Intrexon
         ['INO',7.0,15.0,'NASDAQ','Inovio Zika?'], # zika indicator stock. Inovio
         ['NLNK',15.0,25.0,'NASDAQ','Newlink Zika?'], # zika indicator stock. Newlink        
         ['CERS',4.0,8.0,'NASDAQ','ceries Zika?'], # zika indicator stock. ceries        
         ['SNY',30.0,50.0,'NYSE','sanofi Zika?'], # zika indicator stock. sanofi. dividend 3.72%
         ['MDVN',40.0,80.0,'NASDAQ','Medivation Zika?'], # zika indicator stock. Medivation. no dividend
         ['JCP',5.0,15.0,'NYSE','JC pennies'], # JC pennies.
         ['DQ',20.0,35.0,'NYSE','Daqo New Energy Corp'], # Daqo New Energy Corp. zacks rated high
         ['CAT',60.0,90.0,'NYSE','Catepillar'], # Catepillar, 3.8% dividend. most shorted
         ['CBA',6.0,9.0,'NYSE','clearbridge. energy company'], # clearbridge. energy company, 10% dividend. 
         ['UTX',80.0,120.0,'NYSE',' united  technolgy. airplane builder'], # united  technolgy. airplane builder most shorted, 2.4% dividend.
         ['HON',95.0,130.0,'NYSE','honeywell'], # honeywell. 2%
         ['V',65.0,100.0,'NYSE','visa'], # visa. 0.7%
         ['MO',50.0,70.0,'NYSE','tobacco Altria'], # tobacco company. Altria 3%
         ['RAI',40.0,60.0,'NYSE','reynolds tobacco'], # reynolds stock. tobacco. 3%
         ['STZ',140.0,180.0,'NYSE','constellation drinks'], # constellation drinks stock. 1%
         ['BWLD',100.0,180.0,'NASDAQ','BW3s'], # BW3's
         ['TXRH',35.0,80.0,'NASDAQ','texas road house'], # texas road house
         ['CGNX',30.0,80.0,'NASDAQ','machine vision'], # machine vision. 0.8%
         ['MBLY',30.0,80.0,'NYSE','mobileye vision based driving'], # mobileye vision based driving
         ['CFX',30.0,80.0,'NYSE','colfax'], # colfax?
         ['PCLN',1000.0,1500.0,'NASDAQ','priceline'], # priceline
         ['TRIP',50.0,70.0,'NASDAQ','trip adviser'], # trip adviser
         ['SWHC',10.0,30.0,'NASDAQ','smith and wessin'], # smith and wessin
         ['RGR',40.0,70.0,'NYSE','ruger'], # ruger 2.5% dividend
         ['OLN',10.0,30.0,'NYSE','winchester++'], # winchester++ 3% dividend
         #['TWLO',20.0,40.0,'NASDAQ'], # twilio
         ['BKS',7.0,15.0,'NYSE','barnes & nobles'], # barnes & nobles. 5% dividend
         ['DNKN',35.0,55.0,'NASDAQ','dunkin doughnuts'], # dunkin doughnuts. 2.7% dividend         
         ['SBUX',35.0,75.0,'NASDAQ','starbucks'], # starbucks. 1.5% dividend         
    #['KKD',15.0,30.0,'NYSE','krispy kreme'], # krispy kreme 
         ['JVA',4.0,10.0,'NASDAQ','pure coffee holding'], # JAVA. pure coffee holding
         ['VIAB',30.0,80.0,'NASDAQ','viacom'], # viacom 3.7% dividend

         ['^DJI',17.0e3,22.0e3,'NYSE','DJIA'], # DJIA
         ['XTN',30.0,80.0,'NYSE','S&P transport'], # S&P transport
         ['DJTA',7.0e3,10.0e3,'NYSE','DJIA transport'], # DJIA transport
    ['IYT',100.0,200.0,'NYSEARCA','ishare DJIA transport'], # ishare DJIA transport    
         ['CSX',20.0,60.0,'NASDAQ','Train manufacture'], # Train manufacture. 1.6%
         ['SB',1.0,2.0,'NYSE','safe builder'], # safe builder. 2.0% 
         ['VIOO',60.0,150.0,'NYSE','small cap'], # small cap
         ['MDY',200.0,300.0,'NYSE','mid cap'], # mid cap
         ['GS',150.0,300.0,'NYSE','Goldman saks'], # Goldman saks. 1% dividend
            ['FAF',20.0,70.0,'NYSE','investment corp'], # investment. 3.5.% dividend        
         ['JPM',65.0,120.0,'NYSE','JPM chase'], # JPM chase. 2% dividend
         ['PNC',90.0,150.0,'NYSE','PNC bank'], # PNC bank. 2% dividend
         ['ADS',90.0,350.0,'NYSE','alliance data systems'], # alliance data systems 0.9%
         ['C',20.0,350.0,'NYSE','citigroup'], # citigroup 1.06%
         ['USB',20.0,350.0,'NYSE','us bancorp'], # us bancorp 2%
         ['VGT',90.0,150.0,'NYSEARCA','Vanguard information tech'], # Vanguard information tech. 1.4% dividend
    ['^RUT',900.0,1500.0,'INDEXRUSSELL','russel 200'], # russel 2000
    ['^RUA',900.0,1500.0,'INDEXRUSSELL','russel 3000'], # russel 3000
    ['^RUI',900.0,1500.0,'INDEXRUSSELL','russel 1000 growth index'], # russel 1000 growth index
    ['IWS',30.0,500.0,'NYSEARCA','russel 2000'], # russel 2000. 1.7% dividend
    ['IWM',30.0,500.0,'NYSEARCA','russel midcaps'], # russel midcaps. 2.3% dividend
    ['IWO',30.0,500.0,'NYSEARCA','russel 2000 growth index'], # russel 2000 growth index. 1.2% dividend
    ['IWN',30.0,500.0,'NYSEARCA','russel 2000 value index'], # russel 2000 value index. 2.3% dividend
    ['IWB',30.0,500.0,'NYSEARCA','russel 1000 index'], # russel 1000 index. 2.4% dividend
    ['IWL',30.0,500.0,'NYSEARCA','russel top 200'], # russel top 200 1.88% dividend
    ['IWF',30.0,500.0,'NYSEARCA','russel 1000 growth index'], # russel 1000 growth index. 1.8% dividend
    ['^VIX',0.0,20.0,'INDEXCBOE','Volatility Index'], # Volatility Index. look for a point where the price is 10 points from the MA. spikes indicate fear
    ['EL',30.0,500.0,'NYSE','estee lauder'], # estee lauder

         #['NTDOY',30.0,80.0,'OTCMKTS'], # viacom 3.7% dividend         
        #['SPY',60.0,90.0], # spyder large cap mutual fund
        #['VIG',60.0,90.0], # vanguard large cap mutual fund 3.1% dividend
        #['WTI',20.0,35.0], # west texas intermediate. crude oil
        #['NDX',2000.0,5000.0], # nasdaq index  
        ]
etfs = [['SPY',8.0,20.0,'NYSE','SPY'], 
            ['QQQ',8.0,20.0,'NYSE','nasdaq'], 
            ['XLE',8.0,20.0,'NYSE','Energy Select Sector SPDR Fund'], 
            ['XLF',8.0,20.0,'NYSE','Financial Select Sector SPDR Fund'],
            ['XLU',8.0,20.0,'NYSE','Utilities Select Sector SPDR Fund'],
            ['XLI',8.0,20.0,'NYSE','Industrial Select Sector SPDR Fund'],
            ['GDX',8.0,20.0,'NYSE','VanEck Vectors Gold Miners ETF'],
            ['XLK',8.0,20.0,'NYSE','Technology Select Sector SPDR Fund'],
            ['XLV',8.0,20.0,'NYSE','Health Care Select Sector SPDR Fund'],
            ['XLY',8.0,20.0,'NYSE','Consumer Discretionary Select Sector SPDR Fund'],
            ['XLP',8.0,20.0,'NYSE','Consumer Staples Select Sector SPDR Fund'],
            ['XLB',8.0,20.0,'NYSE','Materials Select Sector SPDR Fund'],
            ['XOP',8.0,20.0,'NYSE','Spdr S&P Oil & Gas Exploration & Production Etf'],
            ['IYR',8.0,20.0,'NYSE','iShares U.S. Real Estate ETF'],
            ['XHB',8.0,20.0,'NYSE','Spdr S&P Homebuilders Etf'],
            ['ITB',8.0,20.0,'NYSE','iShares U.S. Home Construction ETF'],
            ['VNQ',8.0,20.0,'NYSE','Vanguard Real Estate Index Fund ETF Shares'],
            ['GDXJ',8.0,20.0,'NYSE','VanEck Vectors Junior Gold Miners ETF'],
            ['IYE',8.0,20.0,'NYSE','iShares U.S. Energy ETF'],
            ['OIH',8.0,20.0,'NYSE','VanEck Vectors Oil Services ETF'],
            ['XME',8.0,20.0,'NYSE','SPDR S&P Metals & Mining ETF'],
            ['XRT',8.0,20.0,'NYSE','Spdr S&P Retail Etf'],
            ['SMH',8.0,20.0,'NYSE','VanEck Vectors Semiconductor ETF'],
            ['IBB',8.0,20.0,'NYSE','iShares Nasdaq Biotechnology ETF'],
            ['KBE',8.0,20.0,'NYSE','SPDR S&P Bank ETF'],
            ['KRE',8.0,20.0,'NYSE','SPDR S&P Regional Banking ETF'],
            ['XTL',8.0,20.0,'NYSE','SPDR S&P Telecom ETF'],]
