from ReadData import SQL_CURSOR
import sqlite3
s=SQL_CURSOR()
sc = s.cursor()

table_names = sc.execute("SELECT name from sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';").fetchall()
for tname in table_names:
    #print(tname[0])
    if tname[0].count('-'):
        continue
    try:
        #print(sc.execute('SELECT MIN(rowid) from SPY GROUP BY Date').fetchall())
        distin = sc.execute('SELECT COUNT(DISTINCT Date) from %s' %tname[0]).fetchall()[0][0]
        allD = sc.execute('SELECT COUNT(Date) from  %s' %tname[0]).fetchall()[0][0]
        if abs(allD-distin)>0:
            print(distin,allD,tname[0])
            #print(sc.execute('SELECT Date from  %s' %tname[0]).fetchall())
            print('DELETE FROM %s WHERE rowid NOT IN ( SELECT MIN(rowid) from  %s GROUP BY Date)' %(tname[0],tname[0]))
            #sc.execute('DELETE FROM %s WHERE rowid NOT IN ( SELECT MIN(rowid) from  %s GROUP BY Date)' %(tname[0],tname[0]))
            sc.execute('DROP TABLE %s'%(tname[0] ))
            #print(sc.execute('SELECT COUNT( Date) from %s' %tname[0]).fetchall()[0][0])
    except sqlite3.OperationalError:
        print('Could not load!')
sc.close()
#DELETE FROM lipo
#WHERE rowid NOT IN (
#  SELECT MIN(rowid) 
#  FROM lipo 
#  GROUP BY messdatum
#)
