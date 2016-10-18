import pandas as pd;
import time;
import sqlite3;
import numpy as np;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')

#tiermode='TRTIER2'
tiermode='TRCODE'

print("processing...")

trans = pd.DataFrame()
matrix = pd.DataFrame()

trans['case'] = jointable['TUCASEID']
trans['caseshift'] = jointable['TUCASEID'].shift(-1)
#trans['step'] = jointable['TUACTIVITY_N']
trans['day'] = jointable['TUDIARYDAY']
trans['hour'] = jointable['TUCUMDUR24'].apply(lambda x: np.floor(x/60.0))
trans['origin'] = jointable[tiermode]
trans['dest'] = jointable[tiermode].shift(-1)
#trans['corigin'] = jointable.apply((lambda x: (x['TUCC5'] == 1) or (x['TUCC5B'] == 1) or (x['TUCC7'] == 1) or (x['TUCC8'] == 1)),axis=1)
#trans['cdest'] = trans['corigin'].shift(-1)

trans = trans[trans.caseshift.notnull()]

trans['caseshift'] = trans['caseshift'].apply(lambda x:int(x))
trans['dest'] = trans['dest'].apply(lambda x:int(x))

trans = trans[trans.case == trans.caseshift]
trans.drop('caseshift',axis=1,inplace =True)

#trans.to_csv(datapath + "transitions.csv");

#print(len(set(trans['dest'])));
g = trans.groupby(['origin','dest','day','hour'])
tmap = pd.DataFrame(list(g.indices),columns=['origin','dest','day','hour'])
tmap['count'] = list(g.size())

g = tmap.groupby(['day','hour'])

tmap['prob'] = g['count'].apply(lambda x: x / x.sum())
#tcount = len(tmap);
#print(tmap);
#print(tmap)
#print(len(tmap));
# 
# steps = 7 * 24;
# 
# data = np.zeros((steps,tcount));


con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/transprob.sq3");
tmap.to_sql('transprob',con);
con.close();


