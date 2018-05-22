import h5py
import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.stats import linregress;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
infile = h5py.File(datapath +"finalexptraj1.h5",'r')
ds = infile["/exptraj"][:]
ds = ds.reshape(1000,96)
ds = np.clip(ds,0.0,np.max(ds))
# ds = np.cumsum(ds,axis=1)


for i in ds[:100]:
	plt.plot(i,linewidth=0.5,alpha=0.5,color='k')
plt.axes().set_xticks((0,24,48,72,96))
plt.axes().set_xticklabels(("4:00", "10:00", "16:00", "22:00","4:00"))
plt.axes().set_yticks((10000,20000,30000,40000))
plt.axes().set_yticklabels((10,20,30,40))
plt.show()

sums = np.sum(ds,axis=1);
maxvals = np.max(ds,axis=1)
means = np.mean(ds,axis=1)
# mat = np.array([sums,means,maxvals]).T
# mat = ds;

# plt.scatter(sums,maxvals,s=5.0);
# plt.show()

# tsn = TSNE(early_exaggeration=50.0,perplexity=50,verbose=5)
# fit = tsn.fit_transform(ds)
# nfit = fit / np.max(np.abs(fit));
# dbs = DBSCAN(eps = 0.03, min_samples=2)
# labels = dbs.fit_predict(nfit);

# plt.scatter(nfit.T[0],nfit.T[1],s=5,c=labels)
# plt.show()

# sortorder = np.argsort(labels)

sortorder = np.argsort(np.percentile(ds,q=50.0,axis=1))
# sortorder = np.argsort(sums)
sortds = ds[sortorder]
sortsum = sums[sortorder]

# plt.pcolormesh(sortds.T)
# plt.show()

# lr = linregress(sums,maxvals);
# print(lr[2],lr[2]**2,lr[3])


ax1 = plt.subplot(111)
percentiles = [5.0,50.0,95.0]
for i in percentiles:
	ax1.plot(np.percentile(sortds,q=i,axis=1),linewidth=0.5)
ax1.plot((0,1000),(0,0),color='black',linewidth=0.5)

ax2 = ax1.twinx()
ax2.plot(sortsum, color='black',linewidth=0.5)

plt.show()

# lset = set(labels);
# print(lset)
# for i in lset:                          
# #a = np.mean(ds[labels == i],axis=0)                  
# 	for j in ds[labels == i]:
# 		plt.plot(j,linewidth=0.5);
# 	plt.show()

