import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import h5py
import pandas as pd
from scipy.stats import gaussian_kde;
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter;
from scipy.stats import linregress
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from collections import Counter;

def plthist(ax,arr1,arr2,alpha=1.0,color='RdBu'):
	hist,xedge,yedge = np.histogram2d(arr1,arr2,bins=[200,200])#,range=[(0.0,1.0),[0.0,np.max(arr2)]] )
	# hist,xedge,yedge = np.histogram2d(arr1/np.max(arr1),arr2/np.max(arr2),bins=[200,200])#,range=[(0.0,1.0),[0.0,np.max(arr2)]] )

	X,Y = np.meshgrid(xedge,yedge)
	hist[hist > 0.0] += (np.max(hist) - hist[hist > 0.0])*0.1
	# ax.set_xlim(0.,1.);
	# ax.set_ylim(0.,1.);
	ax.pcolormesh(X,Y,hist,cmap=color,alpha=alpha)
	# plt.show()



def proctraj(intraj):

	traj = np.reshape(intraj,(100000,96))
	traj = np.clip(traj,0,np.max(traj))


	total = np.sum(traj,axis=1)
	peak = np.max(traj,axis=1)
	acute = 1.0 - np.sum(traj / peak[:,None] / 96.0,axis=1)
	# diffs = np.abs(np.diff(traj,axis=1))
	diffs = np.diff(traj,axis=1)
	absdiffs = np.abs(diffs)
	# diffs = np.abs(np.diff(diffs,axis=1))
	quantile = np.percentile(absdiffs.flatten(),90.0)
	print(quantile)
	# h,e = np.histogram(diffs,bins='auto')
	# plt.plot(e[:-1],h)
	# plt.show()
	highexc = np.copy(absdiffs)
	lowexc = np.copy(absdiffs)

	highexc[absdiffs < quantile] = 0.0
	lowexc[absdiffs >= quantile] = 0.0

	# for i in range(0,100):
	# 	plt.plot(traj[i])
	# 	plt.plot(highexc[i])
	# 	plt.plot(lowexc[i])
	# 	plt.show()


	var = np.var(diffs,axis=1)

	delta = np.sum(diffs,axis=1)


	de = np.median(delta)

	hdelta = np.max(highexc,axis=1) - np.min(highexc,axis=1)
	ldelta = np.max(lowexc,axis=1) - np.min(lowexc,axis=1)

	#number of deltas excursions above/below quantile
	highexc = np.sum(highexc,axis=1) / quantile
	lowexc = np.sum(lowexc,axis=1) / quantile

	# highexc /= np.median(highexc)
	# lowexc /= np.median(lowexc)
	
	highexc /= hdelta
	lowexc /= ldelta
	


	# inq=np.percentile(traj,90.0,axis=1)-np.percentile(traj,10.0,axis=1)


	f = lambda x: (np.min(x),np.max(x))
	print("T:",f(total))
	print("P:",f(peak))
	print("A:",f(acute))
	print("D:",f(delta))
	
	#  plt.scatter(acute,np.log10(delta))
	# plt.show()

	lr = linregress(delta,total); print("D,T:",lr.rvalue, lr.rvalue**2)
	lr = linregress(delta,peak); print("D,P:",lr.rvalue, lr.rvalue**2)
	lr = linregress(delta,acute); print("D,A:",lr.rvalue, lr.rvalue**2)
	lr = linregress(acute,total); print("A,T:",lr.rvalue, lr.rvalue**2)
	lr = linregress(acute,peak); print("A,P:",lr.rvalue, lr.rvalue**2)
	lr = linregress(peak,total); print("P,T:",lr.rvalue, lr.rvalue**2)

	return np.array([total,acute,highexc,lowexc]),traj;


# plthist(acute,delta);
# plthist(acute,total);
# plthist(acute,peak); 

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run03/"
exptraj0 = h5py.File(datapath + "finalexptraj-0.h5",'r')
exptraj1 = h5py.File(datapath + "finalexptraj-1.h5",'r')
exptraj2 = h5py.File(datapath + "finalexptraj-2.h5",'r')
proctraj0,traj0 = proctraj(exptraj0["/exptraj"][:])
proctraj1,traj1 = proctraj(exptraj1["/exptraj"][:])
proctraj2,traj2 = proctraj(exptraj2["/exptraj"][:])
slist = exptraj0["/slist"][:]

proctrajz,trajz = proctraj(traj0+traj1+traj2);

# indvlabel = pd.read_csv("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/indvlabels.csv",index_col=0)
# print(slist[:10])
# labels = indvlabel.iloc[slist]["casetype"].values
# print(labels[:10])

# plt.scatter(proctraj0[0],proctraj0[2],s=1,c=labels)
# plt.show()

def plot3d(ax,arr):
	sample = np.random.choice(len(arr[0]),size=500)
	
	ax.scatter(arr[0][sample],arr[1][sample],zs=arr[3][sample],s=1,alpha=0.5)
	ax.set_xlim(0.0,4000000)
	ax.set_ylim(0.0,100000.0)
	ax.set_zlim(0.0,200000.0)
	
# ax = plt.figure().add_subplot(111,projection='3d')
# plot3d(ax,proctraj0)
# plot3d(ax,proctraj1)
# plot3d(ax,proctraj2)
# plt.show()
# exit()

# limit = 2500

# tsn = TSNE(early_exaggeration=50.0,perplexity=50,verbose=5)
# fit = tsn.fit_transform(proctraj0.T[:limit])
# nfit = fit / np.max(np.abs(fit));


# # for e in np.arange(0.01,0.06,0.01):
# # 	for m in [2,3,5,10]:
# # 		print(e,m)
# e = 0.02
# m = 5
# dbs = DBSCAN(eps = e, min_samples=m)
# labels = dbs.fit_predict(nfit);

# scount = Counter(labels)
# print(list(zip(scount.keys(),scount.values())))

# plt.scatter(nfit.T[0],nfit.T[1],s=5,c=labels)
# plt.show()

# sortorder = np.argsort(labels)
# for i in range(4):
# 	plt.plot((proctraj0[i][:limit][sortorder]/np.max(proctraj0[i][:limit]))+float(i),linewidth=0.5,alpha=0.5)

# plt.show()
# lset = set(labels);
# # print(lset)
# for i in lset:                          
# #a = np.mean(ds[labels == i],axis=0)                  
# 	for j in traj0[:limit][labels == i][:500]:
# 		plt.plot(j,linewidth=0.5);
# 	plt.show()

fontsize=12
arr = [ proctrajz, proctraj0, proctraj1, proctraj2 ]
tarr = [trajz,traj0,traj1,traj2]
labels = [ "Sum","Non-Work","Work","Travel"]
xlabels = ["Exposure, Person-Minutes","Acuteness Factor"]
xtickset = [ [ 0.0, 0.5e7, 1.0e7, 1.5e7, 2.0e7 ], [0.0,0.25,0.5,0.75,1.0] ]
xticklabel = [ [ '0.0', '', '1.0e7', '', '2.0e7' ], ['0.0','0.25','0.5','0.75','1.0'] ]
# xlims = [ (-1.0e3,1.1e7), (0.0,1.0) ]


fig = plt.figure(figsize=(8.0,3.0))

styles = ['-','--','-.',':']
for j in range(0,2):
	ax = fig.add_subplot(1,3,j+2)
	for ind,i in enumerate(arr):
		
		mx = np.max(i[j])
		# k = gaussian_kde(i[j])
		# print(k)
		# x = np.linspace(0.0,mx,1000)
		# plt.plot(x,k.pdf(x),linewidth=1.0,linestyle=':')
		hist,edge = np.histogram(i[j],bins='auto')

		ax.plot(edge[:-1],hist/np.sum(hist),linewidth=0.95,color='k',label=labels[ind],linestyle=styles[ind])
	if(j == 0): ax.legend();
	# ax.set_xlim(*xlims[j])
	ax.set_yticks([])
	ax.set_xlabel(xlabels[j],fontsize=fontsize)
	ax.set_xticks(xtickset[j])
	ax.set_xticklabels(xticklabel[j],fontsize=fontsize)
	# plt.xlim(0.0,1.0)
# plt.legend()

pick = np.random.choice(len(trajz),size=5)
print(pick)
ax = fig.add_subplot(1,3,1)
greyrange = np.linspace(0.80,0.20,5)
for i,g in zip(pick,greyrange):
	ax.plot(trajz[i],linewidth=0.95,color=str(g))


ax.set_xlim(-1,97)
ax.set_xticks([0,24,48,72,96])
ax.set_xticklabels(['4:00','10:00','16:00','22:00','4:00'],fontsize=fontsize)
ax.set_xlabel("Time",fontsize=fontsize)
ax.set_yticks([])


plt.tight_layout();

plt.savefig('exposefactors.eps',dpi = 300)

# plt.show()

exit()


def f(val):
	return '{:0.2f}'.format(val) + " "

for i in range(0,100000):
	for pj,tj in zip(arr,tarr):
		l = f(pj[0][i]) + f(pj[1][i]) + f(pj[2][i]) + f(pj[3][i])
		plt.plot(tj[i],linewidth=0.5,label=l)
	plt.legend();
	plt.show()


# exit()

# thresh = 0.65
# ax = plt.subplot(211)
# smp = traj1[proctraj1[2] > thresh]
# sample = np.random.choice(len(smp),size=500)
# for i in smp[sample]:
# 	ax.plot(i,linewidth=0.1,alpha=0.5,c='k')
# ax = plt.subplot(212)
# smp = traj1[proctraj1[2] < thresh]
# sample = np.random.choice(len(smp),size=500)
# for i in smp[sample]:
# 	ax.plot(i,linewidth=0.1,alpha=0.5,c='k')
# plt.show()


pairs = [ (0,1),(0,3),(1,3),(2,0), (2,1), (2,3) ]

for i in pairs:
	ax = plt.subplot(131)
	plthist(ax,proctraj0[i[0]],(proctraj0[i[1]]),alpha=1.0,color='Greys')
	ax = plt.subplot(132)
	plthist(ax,proctraj1[i[0]],(proctraj1[i[1]]),alpha=1.0,color='Greys')
	ax = plt.subplot(133)
	plthist(ax,proctraj2[i[0]],(proctraj2[i[1]]),alpha=1.0,color='Greys')
	plt.show()

exit()
# traj = exptraj["/exptraj"][:]
# traj = np.reshape(traj,(100000,96))
# traj = np.clip(traj,0,np.max(traj))



# for i in range(0,200,5):
#     for j in range(0,5):
#         plt.plot(traj[i+j],linewidth=0.5,label=acute[i+j])
#     plt.legend();
#     plt.show()


# for i in range(0,200,5):
#     for j in range(0,5):
#         plt.plot(traj[i+j],linewidth=0.5,label=acute[i+j])
#     plt.legend();
#     plt.show()



total = np.sum(traj,axis=1)
peak = np.max(traj,axis=1)
acute = 1.0 - np.sum(traj / peak[:,None] / 96.0,axis=1)
delta = np.sum(np.abs(np.diff(traj,axis=1)),axis=1)

sortorder = np.argsort(acute)

lr = linregress(delta,total); print("D,T:",lr.rvalue, lr.rvalue**2)
lr = linregress(delta,peak); print("D,P:",lr.rvalue, lr.rvalue**2)
lr = linregress(delta,acute); print("D,A:",lr.rvalue, lr.rvalue**2)
lr = linregress(acute,total); print("A,T:",lr.rvalue, lr.rvalue**2)
lr = linregress(acute,peak); print("A,P:",lr.rvalue, lr.rvalue**2)
lr = linregress(peak,total); print("P,T:",lr.rvalue, lr.rvalue**2)


plt.plot(delta[sortorder]/np.max(delta),label='D',linewidth=0.5,alpha=0.5)
plt.plot(acute[sortorder],label='A',linewidth=0.5,alpha=0.5)
plt.plot(peak[sortorder]/np.max(peak),label='P',linewidth=0.5,alpha=0.5)
plt.plot(total[sortorder]/np.max(total),label='T',linewidth=0.5,alpha=0.5)
plt.legend()
plt.show()

# plthist(delta,total);
# plthist(delta,peak); 
plthist(delta,acute);
plthist(acute,total);
plthist(acute,peak); 
# plthist(peak,total); 

def f(val):
	return '{:0.2f}'.format(val) + " "

for i in range(0,200,5):
	for j in range(0,5):
		l = f(peak[i+j])+f(acute[i+j])+f(total[i+j])+f(delta[i+j])
		plt.plot(traj[i+j],linewidth=0.5,label=l)
	plt.legend();
	plt.show()


# plt.scatter(delta,peak,s=1,alpha=0.5); plt.show()
# plt.scatter(acute,total),s=1,alpha=0.5); plt.show()
# plt.scatter(acute,total,s=1,alpha=0.5); plt.show()




# import h5py
# import matplotlib.pyplot as plt;
# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN
# from scipy.stats import linregress;
# from collections import Counter;

# datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
# datapath = '/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run03/'
# infile = h5py.File(datapath +"finalexptraj-2.h5",'r')
# ds = infile["/exptraj"][:]
# ds = ds.reshape(100000,96)
# ds = np.clip(ds,0.0,np.max(ds))
# # ds = np.cumsum(ds,axis=1)


# rcutoff = 0.75
# maxc = 50

# assign = np.full(len(ds),-1)
# inds = [0,]
# assign[0] = 0;
# for i in range(1,len(ds)):
# 	if(i % 100) == 0: 
# 		print(i, len(inds),end=',',flush=True)
# 	rmax = 0.0; rind = 0;
# 	for j in inds:
# 		reg = linregress(ds[i],ds[j])
# 		if(reg.rvalue > rcutoff):
# 			assign[i] = j;
# 			break;
# 		if(reg.rvalue > rmax):
# 			rmax = reg.rvalue;
# 			rind = j;
# 	if(assign[i] == -1): 
# 		if(len(inds) < maxc):
# 			assign[i] = i;
# 			inds += [i]
# 		else:
# 			assign[i] = rind


# scount = Counter(assign)
# print(list(zip(scount.keys(),scount.values())))

# labels = assign;

# # for i in ds[:1000]:
# # 	plt.plot(i,linewidth=0.5,alpha=0.5,color='k')
# # plt.axes().set_xticks((0,24,48,72,96))
# # plt.axes().set_xticklabels(("4:00", "10:00", "16:00", "22:00","4:00"))
# # plt.axes().set_yticks((10000,20000,30000,40000))
# # plt.axes().set_yticklabels((10,20,30,40))
# # plt.show()

# sums = np.sum(ds,axis=1);
# maxvals = np.max(ds,axis=1)
# means = np.mean(ds,axis=1)
# # mat = np.array([sums,means,maxvals]).T
# # mat = ds;

# # plt.scatter(sums,maxvals,s=5.0);
# # plt.show()

# # tsn = TSNE(early_exaggeration=50.0,perplexity=50,verbose=5)
# # fit = tsn.fit_transform(ds[:5000])
# # nfit = fit / np.max(np.abs(fit));
# # dbs = DBSCAN(eps = 0.02, min_samples=5)
# # labels = dbs.fit_predict(nfit);

# # plt.scatter(nfit.T[0],nfit.T[1],s=5,c=labels)
# # plt.show()

# sortorder = np.argsort(labels)

# # sortorder = np.argsort(np.percentile(ds,q=50.0,axis=1))
# # sortorder = np.argsort(sums)
# sortds = ds[sortorder]
# sortsum = sums[sortorder]

# # plt.pcolormesh(sortds.T)
# # plt.show()

# # lr = linregress(sums,maxvals);
# # print(lr[2],lr[2]**2,lr[3])


# ax1 = plt.subplot(111)
# percentiles = [5.0,50.0,95.0]
# for i in percentiles:
# 	ax1.plot(np.percentile(sortds,q=i,axis=1),linewidth=0.5)
# ax1.plot((0,1000),(0,0),color='black',linewidth=0.5)

# ax2 = ax1.twinx()
# ax2.plot(sortsum, color='black',linewidth=0.5)

# plt.show()

# lset = set(labels);
# # print(lset)
# for i in lset:                          
# #a = np.mean(ds[labels == i],axis=0)                  
# 	for j in ds[labels == i][:500]:
# 		plt.plot(j,linewidth=0.5);
# 	plt.show()

