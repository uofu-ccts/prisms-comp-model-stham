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

pd.set_option('display.max_rows', 10)
np.set_printoptions(threshold=np.inf)

sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/mathmodel/python")
import actprofile as ap;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"
outpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/test/"
imgpath = outpath + "singleact-" + time.strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(imgpath)

print("loading...")

acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
acttable = pd.merge(acttable,infotable[['TUCASEID','TUDIARYDAY']],on='TUCASEID')

actmapping = np.sort(list(set(acttable['TRCODE'])))

ati = { tr:i for i,tr in enumerate(actmapping) }
ita = { i:tr for i,tr in enumerate(actmapping) }

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
print(wti)

acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
acttable['end'] = acttable['TUCUMDUR24']
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);
acttable['whereind'] = acttable['TEWHERE']#.apply(lambda x: wti[x]);
acttable['day'] = acttable['TUDIARYDAY']

acttable = acttable[['actind','start','end','whereind']]
acttable.info()

print("processing...")

actgroup = acttable.groupby('actind')

for g,df in actgroup:
	if(len(df) < 10): continue;
	if(len(df) > 5000): continue;

	# if(g not in [ati[10102],ati[20101]]): continue

	print("Group: ",g);
	initclf, initlabels = ap.randoTrees(df.values,depth=3,nest=500)
	#prox = ap.proxMat(initlabels,procs=5);
	#coords, dblabels = ap.tsneLabelFit(prox, eps=0.5, samples = samples,perplex=perplex);



	# iso = sklearn.manifold.Isomap(n_components=2,n_jobs=6)
	# coords = iso.fit_transform(df.values).T

	pca = sklearn.decomposition.TruncatedSVD(n_components=4)
	truncated = pca.fit_transform(initlabels)

	tsne = sklearn.manifold.TSNE(n_components=2,verbose=2)
	coords = tsne.fit_transform(truncated)

	dbscan = sklearn.cluster.DBSCAN(eps = 1.0, min_samples=2);
	dblabels = dbscan.fit_predict(coords)

	ap.compPlot(coords.T,dblabels,imgpath+"/act-"+str(ita[g]));


	cmapp = np.linspace(0.0,1.0,len(set(dblabels)));
	colors = [ cm.jet(x) for x in cmapp ]
	# np.random.shuffle(colors);
	outc = [ colors[b] if b > -1 else (0,0,0,1) for b in dblabels ];
	plt.scatter(df['start'].values,df['end'].values,color=outc,s=8,alpha=0.5,edgecolor='')

	# coords = coords.T;

	# #coords = df[['start','whereind']].values.T;
	# plt.scatter(coords[0],coords[1],s=5,edgecolor="none",alpha=0.5)
	plt.title("Start/END Act: "+str(ita[g])+", Count: "+str(len(df)))
	F = plt.gcf()
	F.set_size_inches(8,6)
	F.set_dpi(300);
	F.savefig(imgpath+"/actstend-"+str(ita[g])+".png");
	plt.clf()
	# ap.dbscangrid(coords.T,imgpath+"/act-"+str(g),eps=[0.1,1.0,5.0]);

	# plt.scatter(df['start'].values,df['day'].values,color=outc,s=8,alpha=0.5,edgecolor='')
	# plt.title("Start/Day Act: "+str(ita[g])+", Count: "+str(len(df)))
	# F = plt.gcf()
	# F.set_size_inches(8,6)
	# F.set_dpi(300);
	# F.savefig(imgpath+"/actstwh-"+str(ita[g])+".png");
	# plt.clf()




