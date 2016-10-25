import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;

sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/test")
import bettersankey as bsk;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')

#tiermode='TRTIER2'
tiermode='TRCODE'

#columns=['case','day','hour','origin','dest','corigin','cdest']
trans = pd.DataFrame();

print("processing...")

trans['case'] = jointable['TUCASEID']
trans['caseshift'] = jointable['TUCASEID'].shift(-1)
trans['step'] = jointable['TUACTIVITY_N']
trans['day'] = jointable['TUDIARYDAY']
trans['hour'] = jointable['TUCUMDUR24'].apply(lambda x: np.floor(x/60.0))
trans['origin'] = jointable[tiermode]
trans['dest'] = jointable[tiermode].shift(-1)
trans['corigin'] = jointable.apply((lambda x: (x['TUCC5'] == 1) or (x['TUCC5B'] == 1) or (x['TUCC7'] == 1) or (x['TUCC8'] == 1)),axis=1)
trans['cdest'] = trans['corigin'].shift(-1)

trans = trans[trans.caseshift.notnull()]

trans['caseshift'] = trans['caseshift'].apply(lambda x:int(x))
trans['dest'] = trans['dest'].apply(lambda x:int(x))

trans = trans[trans.case == trans.caseshift]
trans.drop('caseshift',axis=1,inplace =True)

trans.to_csv(datapath + "transitions.csv");

print(len(set(trans['dest'])));

s = trans.groupby(['origin','dest']).size()

s.to_csv(datapath + "transitioncounts.csv")

print("plotting...")

v = s.unstack().as_matrix();
v[np.isnan(v)] = 0.0;
logv = np.log10(v);
logv[np.isneginf(logv)] = 0.0;

print("Max value:", np.max(v), " (",np.max(logv),")")

plt.pcolormesh(logv,cmap='Blues');
plt.colorbar();
plt.yticks(np.arange(0,len(s.index.levels[0]),1),s.index.levels[0])
plt.xticks(np.arange(0,len(s.index.levels[0]),1),s.index.levels[0],rotation=45);

plt.show()

exit();
	

