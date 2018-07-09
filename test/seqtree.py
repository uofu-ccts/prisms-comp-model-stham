import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;
import sklearn.decomposition
import sklearn.cluster
import sklearn.ensemble
import sklearn.tree
from sklearn.externals import joblib;
import pydotplus;
from multiprocessing import Pool;
import h5py;

from collections import Counter;
from datetime import datetime;
import re;
import os;
import sys;

pd.set_option('display.max_rows', 50)
# np.set_printoptions(threshold=np.inf)

# sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/mathmodel/python")
# import actprofile as ap;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"
outpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/test/"
# imgpath = outpath + "singleact-" + time.strftime("%Y-%m-%d_%H-%M-%S")
# os.mkdir(imgpath)


print("loading...")

acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
# infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
# acttable = pd.merge(acttable,infotable[['TUCASEID','TUDIARYDAY']],on='TUCASEID')

actmapping = np.sort(list(set(acttable['TRCODE'])))

ati = { tr:i for i,tr in enumerate(actmapping) }
ita = { i:tr for i,tr in enumerate(actmapping) }

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
print(wti)

acttable['start'] = np.floor((acttable['TUCUMDUR24']-acttable['TUACTDUR24'])/60.0)
acttable['end'] = np.floor(acttable['TUCUMDUR24']/60.0)
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);
acttable['whereind'] = acttable['TEWHERE'].apply(lambda x: wti[x]);
# acttable['day'] = acttable['TUDIARYDAY']

acttable = acttable[['TUCASEID','actind','start','end','whereind']]
acttable.info()

# print(acttable);


# shiftframe = acttable['TUCASEID']


frameshift = 3


# seqs = []

# print("seq building...")

# c = 0;
# for g,df in acttable.groupby('TUCASEID'):
# 	if(c % 100 == 0): 
# 		print(c,end=' ')
# 		sys.stdout.flush()

# 	# print(df)

# 	df = df.reset_index(drop=True)
# 	df = df.reindex(index=np.arange(-frameshift,len(df)+frameshift),fill_value=-1.0).reset_index(drop=True).drop('TUCASEID',axis=1)

# 	# print(df)

# 	shiftframe = pd.DataFrame();

# 	for i in range(0,-frameshift-1,-1):
# 		# print(i)
# 		reframe = df.shift(i)
# 		reframe.columns=['actind-'+str(-i),'start-'+str(-i),'end-'+str(-i),'whereind-'+str(-i)]
# 		if(len(shiftframe) == 0): shiftframe = reframe
# 		else: shiftframe = pd.concat([shiftframe,reframe],axis=1,join="inner")

# 	shiftframe = shiftframe[:-frameshift];

# 	shiftframe['caseid'] = g;
	
# 	seqs += [shiftframe]
# 	c+=1;
# 	# if(c > 2): break;

# print("concat...")
# combseqs = pd.concat(seqs,axis=0).reset_index(drop=True)

# # print(combseqs)
# combseqs.info()

combseqs = pd.read_csv(outpath+"seqtree.csv")
combseqs.info()


print("fitting...")
clf = sklearn.ensemble.ExtraTreesClassifier(n_jobs=1,max_leaf_nodes=None,n_estimators=100,criterion='entropy',min_samples_split=2,min_samples_leaf=1,max_depth=10);

vector = combseqs.ix[:,0:frameshift*4].values
labels = combseqs.ix[:,frameshift*4:(frameshift+1)*4].values

# print(vector)
# print(labels)

clf = clf.fit(vector,labels);

print("writing...")
# combseqs.to_csv(outpath+"seqtree.csv")
joblib.dump(clf, outpath+"/clfsave-seqtree.pkl", compress=('gzip',9));