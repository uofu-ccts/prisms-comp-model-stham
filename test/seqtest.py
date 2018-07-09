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
print([(i,k) for i,k in enumerate(actmapping)]);

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
print([(i,k) for i,k in enumerate(locmapping)]);

clf = joblib.load(outpath+"/clfsave-seqtree.pkl")





def seqbuild(clf, seqlength=3):

	seq = -1.0*np.ones(4*seqlength);
	#print(seq)
	
	print(clf.classes_)

	while True:
		pr = clf.predict_proba([seq[-4*seqlength:]])
		print(pr)
		predict = [-1,-1,-1,-1]
		for i in range(4):
			# print(clf.classes_[i],pr[i])
			predict[i] = np.random.choice(clf.classes_[i],p=pr[i][0])
		print(predict)
		seq = np.concatenate([seq,predict],axis=None)
		#print(seq)
		if predict[0] == -1.0:
			break;

	

	return seq;

print("testing...")
print(seqbuild(clf))