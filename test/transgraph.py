import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import scipy;
import scipy.stats
import numpy as np;
import sys;

sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/test")
import bettersankey as bsk;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')


print("processing...");

intervals = 24.0
intsize = 1440.0 / intervals;

lintervals = 1440.0
lintsize = 1440.0 / lintervals

#get set of unique TUCODES
mapping = np.sort(list(set(jointable['TRCODE'])))
count = len(mapping)

tri = { tr:i for i,tr in enumerate(mapping) }
itr = { i:tr for i,tr in enumerate(mapping) }

data = np.zeros((count,7*intervals))
trans = np.zeros((count*count,7*intervals));

g = jointable.groupby(['TUCASEID']);
groups = list(g.groups);


tlengths = np.zeros((count,lintervals + 1));
sleeplengths = np.zeros((intervals, lintervals));



for i in groups:
	code = list(g.get_group(i)['TRCODE']);
	time = list(g.get_group(i)['TUCUMDUR24']);
	acttime = list(g.get_group(i)['TUACTDUR']);
	diaryday = list(g.get_group(i)['TUDIARYDAY'])[0] - 1;
	  
	
	
	
	k = 0;
	baseind = intervals * diaryday;
	for j in range(len(code)):
		data[tri[code[j]]][baseind + k] += 1
		if(tri[code[j]] == 0):
			sleeplengths[k][np.floor(acttime[j]/lintsize)] += 1;
		k = np.floor(time[j]/intsize);
		tlengths[tri[code[j]]][np.floor(acttime[j]/lintsize)] += 1;

	
	for j in range(len(code) - 1):
		trans[tri[code[j]]*count + tri[code[j+1]]][(np.floor(time[j]/intsize) + baseind)] += 1;
		#data[tri[code[j]]][(k + baseind):(np.floor(time[j]/intsize) + baseind)] += 1
		#locdata[k:(np.floor(time[j]/intsize))] = tri[code[j]];
		#trans[tri[code[j]] * count + tri[code[j]]][(k + baseind):(np.floor(time[j]/intsize) + baseind)] += 1
		
	
# 	loctrans[0] = locdata[0]*count + locdata[0];
# 	for j in range(1,int(intervals)):
# 		loctrans[j-1] = locdata[j-1]*count + locdata[j];
# 	for ind, j in enumerate(loctrans):
# 		trans[j][baseind + ind] += 1;
		
	
# 		#origin -> destination
# 			ind = tri[code[j]] * count + tri[code[j+1]]
# 			tslot = np.floor(time[j+1]/intsize) - 1;
# 			trans[ind][(tslot + baseind)] += 1;


# g = jointable.groupby(['TRCODE'])






			
cutoff = 0.025;

print("normalizing...")
data = data.T;
trans = trans.T;

toplist = np.empty(0);

for i in range(len(data)):
	sum = np.sum(data[i]);
	data[i] /= sum;
	sum2 = np.sum(trans[i]);
	trans[i] /= sum2;
	toplist = np.union1d(np.where(trans[i] > cutoff)[0], toplist)
	
data = data.T;
trans = trans.T;

slist = np.zeros_like(toplist);
elist = np.zeros_like(toplist);

for i in range(len(toplist)):
	slist[i] = np.floor(toplist[i]/count); 
	elist[i] = toplist[i] % count;

#print(slist,elist)
toplist = np.stack((slist,elist)).T
#print(toplist)
toptop = np.unique(toplist)
#print(toptop)
sizetop = len(toptop)

# for i in range(len(sleeplengths)):
# 	sleeplengths[i] = sleeplengths[i] / np.sum(sleeplengths[i])

print("building sankey nodes...")

keys = [] #np.zeros((sizetop*sizetop*7*intervals,3))
n = 0;
shifter = 10**np.floor(np.log10(7*intervals*10))
for i in range(sizetop):
	for j in range(sizetop):
		ind = i*count + j;
		for c in range(7*int(intervals)):
			#if(trans[ind][c] > 0.0):
				k = [0,0,0]
				k[0] = np.int(itr[toptop[i]]*shifter + c)
				k[1] = np.int(itr[toptop[j]]*shifter + c + 1)
				k[2] = trans[ind][c];
				keys += [k]
				n += 1;

#print(keys);
#print(len(keys));




print("plotting...")

# #x = np.arange(0.0,7.0,(1.0/intervals))
# cm_subsection = np.linspace(0.0,1.0, count) 
# colors = [ cm.prism(x) for x in cm_subsection ]
# 
# ax = plt.subplot(111)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height])
# ax.set_ylim([0.0,1.0])
# 	
# datasum = np.cumsum(data, axis=0);
# ax.fill_between(x,0,datasum[0],facecolor=colors[0],alpha=0.7,linewidth=0.1)
# for i in range(len(colors)-1):
# 	ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],alpha=0.7,linewidth=0.1)
# print(mapping)
# plt.show();


centiles=[1.,5.,25.,50.,75.,95.,99.]

fit = np.zeros((len(centiles),count));
fit2 = np.zeros((len(centiles),count));
raw = np.zeros((len(centiles),count));
pops = np.zeros(count);
spearman = np.zeros(count);
start = 0.0
stop = 1.0
number_of_lines = count
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ cm.jet(x) for x in cm_subsection ]
c = 0;


x = np.linspace(0,lintervals+1,lintervals+1)
longx = np.linspace(0,lintervals+1.0,1000);
for iind,i in enumerate(tlengths):
	
	if np.sum(i) < 10:
		continue;
	
	a = np.empty(np.sum(i));
	n = 0
	for ind,j in enumerate(i):
		for k in range(int(j)):
			a[n] = ind;
			n+=1;
	
	kern = scipy.stats.gaussian_kde(a)
	w = scipy.stats.exponweib.fit(a,fa=1.0,floc=0.0)
	#wkern = scipy.stats.exponweib.fit(kern(longx),fa=1.0,floc=0.0);
	#p = scipy.stats.exponweib.pdf(x,*w);
	#w2 = scipy.stats.exponweib.fit(a,fc=1.4,floc=0.0)
	#p2 = scipy.stats.exponweib.pdf(x,*w2);
	#pkern = scipy.stats.exponweib.pdf(x,*wkern);
	#pearson[iind] = scipy.stats.spearmanr(kern(x),pkern).correlation;
	#pearson[iind] = scipy.stats.spearmanr(i,p).correlation;
	pops[iind] = len(a);
# 	raw[2][iind] = np.percentile(a,75.0);
# 	raw[1][iind] = np.median(a);
# 	raw[0][iind] = np.percentile(a,25.0);
# 	
# 	#fitintv = scipy.stats.exponweib.interval(0.99,*w)
# 	fit[2][iind] = scipy.stats.exponweib.ppf(0.75, *w)
# 	fit[1][iind] = scipy.stats.exponweib.median(*w);
# 	fit[0][iind] = scipy.stats.exponweib.ppf(0.25, *w)
	for cn in range(len(centiles)):
		raw[cn][iind] = np.percentile(a,centiles[cn])
		fit[cn][iind] = scipy.stats.exponweib.ppf(centiles[cn]/100.0, *w)
	
# 	fitintv = scipy.stats.exponweib.interval(0.99,*w2)
# 	fit2[2][iind] = fitintv[1];
# 	fit2[1][iind] = scipy.stats.exponweib.median(*w2);
# 	fit2[0][iind] = fitintv[0];
	
	#plt.boxplot([a,scipy.stats.exponweib.rvs(*w,size=1000)]);
# 	plt.hist(a, lintervals+1, color='black', alpha=0.3, normed=True);
# 	plt.plot(x,p, color=colors[c],ls='-')
# 	plt.plot(x,p2, color='red',ls='-')
# 
# 	plt.plot((raw[1][iind],raw[1][iind]),(0.0,1.0),'k-')
# 	plt.plot((fit[1][iind],fit[1][iind]),(0.0,1.0),'k--')
# 	plt.plot((fit2[1][iind],fit2[1][iind]),(0.0,1.0),'r-')
# # 	plt.plot(x,pkern, color='black',ls=':')
# 	plt.plot(x,kern(x),color=colors[c],ls='--')
# # 	w1 = scipy.stats.gamma.fit(a)
# 	p = scipy.stats.gamma.pdf(x,*w1);
# 	plt.plot(x,p, color=colors[c],ls='--')
# 	w2 = scipy.stats.foldnorm.fit(a)
# 	p = scipy.stats.foldnorm.pdf(x,*w2);
# 	plt.plot(x,p, color=colors[c],ls=':')
	#print(itr[c]);
	print(itr[iind], len(a), raw[1][iind], fit[1][iind], w) #, w1, w2)
	
	#plt.show();
	c += 1;
	
	#plt.show()



#parg = np.sort(parg);
#parg = np.array(np.arange(len(raw[i])));

#plotmed = median[parg]
#plotfmed = fitmedian[parg]
#plotpop = pops[parg]


fig,ax1 = plt.subplots();

number_of_lines = len(centiles)
cm_subsection = np.linspace(start, stop, number_of_lines) 
colors = [ cm.jet(x) for x in cm_subsection ]
parg = np.argsort(raw[4]);
for i in range(0,len(centiles)):
	
	ax1.errorbar(np.arange(0,len(raw[i])),raw[i][parg],yerr=(fit[i][parg]-raw[i][parg]),fmt='x',color=colors[i]);	

#ax1.errorbar(np.arange(0,len(plotmed)),plotmed,yerr=np.abs(plotfmed-plotmed),fmt='-',color="black");
# ax1.plot(fit[0][parg],'b--')
# ax1.plot(fit[1][parg],'k--')
# ax1.plot(fit[2][parg],'r--')
# 
# ax1.plot(raw[0][parg],'b-');
# ax1.plot(raw[1][parg],'k-');
# ax1.plot(raw[2][parg],'r-');

ax1.set_ylabel("medians")

ax2 = ax1.twinx();
ax2.plot(pops[parg],'g.');
ax2.set_ylabel("pops");

plt.show();


# max = np.max([np.amax(raw),np.amax(fit)])
# ax1.plot((0,max),(0,max),'k-')
# plt.xlim((0,max));
# plt.ylim((0,max));
# 
# 
# for i in range(0,3):
# 	ax1.plot(raw[i],fit[i],styl[i]);
# 
# 
# plt.show()

exit()
# 
# 
# hist,edges = np.histogram(trans,2500,(0.0001,1.0))
# plt.plot(edges[:len(hist)],hist);
# print(np.sum(hist));
# 
# #plt.show()

width = 10.0
height = 10.0
gap = 0.2
pc, text = bsk.patchLayers(keys, width, height, gap);

fig,ax = plt.subplots()
ax.set_axis_off()
ax.margins(0.01)
patch = mpatch.Rectangle((-1.0,-1.0),width+1.0, height+1.0,facecolor='white',edgecolor='none')
ax.add_patch(patch)
for i in pc:
	ax.add_patch(i);
for i in text:
	plt.text(*i, rotation=-45.0);

ax.set_xlim(-1.0, width + 1.0)
ax.set_ylim(-1.0, height + 1.0)

plt.show()

# logtrans = np.log10(trans*10.0+1.0);
# logtrans[np.isneginf(logtrans)] = 0.0;
# 
# plt.pcolormesh(logtrans,cmap='Blues')
# plt.colorbar()
# plt.show();
