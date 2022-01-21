from ReadData import ALPACA_REST,runTicker,ConfigTable,ALPHA_TIMESERIES,GetTimeSlot,SQL_CURSOR
import sys
import sqlite3
ts = ALPHA_TIMESERIES()
sqlcursor = SQL_CURSOR()
sc = sqlcursor.cursor()
doClean=False
doReload=False

ticker='DUG'
daily_prices,j    = ConfigTable(ticker, sqlcursor,ts,'full',hoursdelay=18)
daily_prices_365d = GetTimeSlot(daily_prices, days=365)
split_dates = daily_prices_365d[daily_prices_365d.splitcoef!=1.0]
if len(split_dates)>0:
    print(split_dates)
#print(daily_prices.to_string())
#sc.execute('DROP TABLE DUG')

#list_of_tables = 
table_names = sc.execute("SELECT name from sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';").fetchall()

print("tables: %s" %len(table_names))

#splitID = sc.execute('SELECT COUNT(splitcoef) from DUG WHERE splitcoef!=1.0 AND Date>2021-01-23').fetchall()[0][0]
splitID = sc.execute("SELECT COUNT(splitcoef) from SPY WHERE splitcoef!=1.0 AND Date>'2021-01-23'").fetchall()[0][0]
#splitID = sc.execute("SELECT * from DUG WHERE splitcoef!=1.0 AND Date>'2021-01-23'").fetchall()[0][0]
splitIDl = sc.execute("SELECT * from DUG WHERE splitcoef!=1.0 AND Date>'2021-01-23'").fetchall()[0]
print(splitID)
print(splitIDl)

# create a list to reload
if doReload:
    f=open('reload.txt')
    for a in f:
        print(a)
        daily_prices,j    = ConfigTable(a.strip().strip('\n'), sqlcursor,ts,'full',hoursdelay=18)

#sys.exit(0)
for tname in table_names:
    #print(tname[0])
    if tname[0].count('-'):
        continue
    try:
        #print(sc.execute('SELECT MIN(rowid) from SPY GROUP BY Date').fetchall())
        distin = sc.execute('SELECT COUNT(DISTINCT Date) from %s' %tname[0]).fetchall()[0][0]
        allD = sc.execute('SELECT COUNT(Date) from  %s' %tname[0]).fetchall()[0][0]
        #splitID = sc.execute('SELECT COUNT(splitcoef) from %s' %tname[0]).fetchall()[0][0]
        splitIDinfo = sc.execute("SELECT * from %s WHERE splitcoef!=1.0 AND Date>'2021-01-23'" %tname[0]).fetchall()
        splitIDnum = sc.execute("SELECT COUNT(splitcoef) from %s WHERE splitcoef!=1.0 AND Date>'2021-01-23'" %tname[0]).fetchall()[0][0]
        #print(splitIDnum)
        if splitIDnum>0:
            print('Found a split for ',tname[0],' ',splitIDinfo)

            if doClean:
                print('DROP TABLE %s' %tname[0])                
                sc.execute('DROP TABLE %s'%(tname[0] ))
            #else:
            #    print('Load TABLE %s' %tname[0])
            #    daily_prices,j    = ConfigTable(tname[0], sqlcursor,ts,'full',hoursdelay=18)
            
        if abs(allD-distin)>0:
            print(distin,allD,tname[0])
            #print(sc.execute('SELECT Date from  %s' %tname[0]).fetchall())
            print('DELETE FROM %s WHERE rowid NOT IN ( SELECT MIN(rowid) from  %s GROUP BY Date)' %(tname[0],tname[0]))
            #sc.execute('DELETE FROM %s WHERE rowid NOT IN ( SELECT MIN(rowid) from  %s GROUP BY Date)' %(tname[0],tname[0]))
            if doClean:
                sc.execute('DROP TABLE %s'%(tname[0] ))
            #print(sc.execute('SELECT COUNT( Date) from %s' %tname[0]).fetchall()[0][0])
    except sqlite3.OperationalError:
        print('Could not load!')
sc.close()
