import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
from matplotlib.collections import PatchCollection;
#import scipy.stats.norm as norm;
from scipy.stats import halfnorm, norm, skewnorm;
from scipy.interpolate import Rbf;
from sklearn.mixture import BayesianGaussianMixture;
from sklearn.cluster import SpectralClustering
import h5py;
import numpy as np;
import sys;

pd.set_option('display.max_rows', 2000)
pd.options.mode.chained_assignment = None

def sliceplot(mat):

	ran = np.max(mat[:,2])
	# print(ran);
	cm_subsection = np.linspace(0.0,1.0, ran+1)
	maincolors = [ cm.jet(x) for x in cm_subsection ]
	for i,color in enumerate(maincolors):
		plt.scatter(mat[mat[:,2] == i,0],mat[mat[:,2] == i,1],color=color);
	plt.show()


def breaks1d(df, ncomp=10):
	mat = df.values;
	#print('BayesianGaussianMixture: ')    
	#print(mat)
	bgm = BayesianGaussianMixture(n_components=ncomp,covariance_type='full',max_iter=500,n_init=4).fit(mat);
	pred = bgm.predict(mat);

	return pred,bgm;


def assignWindow(x,wins):
	out = wins[wins['actind']==x['actind']][wins['wmax']>=x['start']][wins['wmin']<=x['start']].index;
	if(len(out) < 1): return -1;
	return out[0];

def assignLen(x,lens):
	out = lens[lens['actind']==x['actind']][lens['lmax']>=x['length']][lens['lmin']<=x['length']].index;
	if(len(out) < 1): return -1;
	return out[0];

def phist(x):
	a,b = np.histogram(x['length'].values,bins='auto');
	a = a/np.sum(a)
	b = b[:-1]
	return np.array([a,b]);

def getwindows(df, ncomp=10):
	ncases = df['instance'].unique().size
	# print(ncases,len(df));
	#print(df)
	allwindows = pd.DataFrame();
	alllengths = pd.DataFrame();

	for i,g in df.groupby(['actind']):
		if len(g) < 10: continue;
		g['window'],winbgm = breaks1d(g[["start"]]);
		g['lenwin'],lenbgm = breaks1d(g[["length"]]);

		# windows = pd.DataFrame();
		# lengths = pd.DataFrame();

		#print( g.groupby('window').apply(lambda x: x['start'].min()) )
		#print( g['window'].value_counts() )

		windows = pd.DataFrame(index=g['window'].unique());
		
		windows['wincount'] = g['window'].value_counts();
		windows['winuniq'] =  g.groupby(['window','instance']).size().reset_index().groupby('window').size();
		windows['density'] = windows['wincount'] / windows['winuniq']
		#print(g)
		#print(windows['wincount'])
		windows['ref'] = windows.index;
		windows['prob'] = windows['winuniq'] / ncases;
		windows['wmin'] = g.groupby('window').apply(lambda x: x['start'].min());
		windows['wmax'] = g.groupby('window').apply(lambda x: x['start'].max());
		windows['wavg'] = g.groupby('window').apply(lambda x: x['start'].mean());
		windows['wstd'] = g.groupby('window').apply(lambda x: x['start'].std());
		windows['wstd'] = windows['wstd'].fillna(1.0);
		windows['actind'] = i;

		allwindows = allwindows.append(windows);

		# print(g['window'].cov(g['lenwin']) )

		lengths =pd.DataFrame(index=g['lenwin'].unique());
		lengths['ref'] = lengths.index;
		lengths['lmin'] = g.groupby('lenwin').apply(lambda x: x['length'].min());
		lengths['lmax'] = g.groupby('lenwin').apply(lambda x: x['length'].max());
		lengths['lavg'] = g.groupby('lenwin').apply(lambda x: x['length'].mean());
		lengths['lstd'] = g.groupby('lenwin').apply(lambda x: x['length'].std());
		lengths['lhist'] = g.groupby('lenwin').apply(phist);


		lengths['lstd'] = lengths['lstd'].fillna(1.0);
		lengths['actind'] = i;



		#print(lengths);
		alllengths = alllengths.append(lengths);

		#print( g.groupby('window').apply(lambda x: x['window'].cov(x['lenwin'])) )
		#print(i,g.groupby(['window']).apply(lambda x: x['lenwin'].value_counts() / x['lenwin'].count() ) );
		#print(i, g.groupby(['instance']).apply(lambda x: tuple(x['window'].sort_values().unique())).value_counts() )


		# print(windows);
	allwindows = allwindows.reset_index();
	allwindows = allwindows.drop("index",axis=1);
	alllengths = alllengths.reset_index();
	alllengths = alllengths.drop("index",axis=1);
	#print(allwindows);
	#print(alllengths);
	# sliceplot(df[['start','length','window']].values);

	return allwindows,alllengths;

def stenmixture(df, ind,ncomp=5):

	#limit = df[df["actind"]==ind]
	matX = df[["start"]].values;
	# print(matX)
	bgm = BayesianGaussianMixture(n_components=ncomp,covariance_type='full',max_iter=500,n_init=3).fit(matX);
	# print(bgm.means_, bgm.covariances_)
	matXY = bgm.predict(matX);



	#bgm = SpectralClustering(n_clusters=ncomp).fit(matX);


	matZ = df[["length"]].values;
	bgm = BayesianGaussianMixture(n_components=ncomp,covariance_type='full',max_iter=500,n_init=3).fit(matZ);
	#print(bgm.means_, bgm.covariances_)
	matXZ = bgm.predict(matZ);

	matXYZ = list(zip(matXY,matXZ))

	s = list(set(matXYZ))
	sti = { tr:i  for i,tr in enumerate(s) };
	matXYZ = np.array([ sti[b] for b in matXYZ ]);

	count =  len(s)
	# print("Count:",count);

	mat = df[["start","length"]].values
	bgm = BayesianGaussianMixture(n_components=count,covariance_type='full',max_iter=500,n_init=10).fit(mat,matXYZ);
	matPred = bgm.predict(mat);
	
	# print(sti,matXYZ)
	# print("Final:", len(list(set(matPred))))


	cm_subsection = np.linspace(0.0,1.0, count)
	maincolors = [ cm.jet(x) for x in cm_subsection ]
	for i,color in enumerate(maincolors):
		plt.scatter(mat[matPred==i,0],mat[matPred==i,1],color=color);
	plt.show();



	#return out;

def demoActPlot(frame,label,actmapping,prefix):

	weekdayset = [2,3,4,5,6]
	#prelim stuff
	#mapping = np.sort(list(set(frame['actind'])))
	actcount = len(actmapping)
	#tri = { tr:i for i,tr in enumerate(mapping) }

	
	#set up coloration
	start = 0.0
	stop = 1.0
	number_of_lines=19
	cm_subsection = np.linspace(start, stop, number_of_lines)
	maincolors = [ cm.prism(x) for x in cm_subsection ]
	colors = []
	for i in actmapping:
		tier1 = i // 10000;
		tier2 = (i // 100) - (tier1 * 100)
		if (tier1 == 50): tier1 = 19;
		if (tier2 == 99): tier2 = 19; 
		tier1 = tier1 - 1;
		tier2 = tier2 - 1
		
		scol = list(maincolors[tier1])
	# 	scol[2] = scol[2] + (tier2 * 0.1)
	# 	if(scol[2] > 1.0): scol[2] = 1.0;
	# 	scol[3] = scol[3] + (tier2 * 0.1)
	# 	if(scol[3] > 1.0): scol[3] = 1.0;
		colors += [tuple(scol)];


	#set up labeling
	legart = []
	leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
	for i in range(number_of_lines):
		legart += [mpatch.Rectangle((0,0),1,1,fc=maincolors[i])]

	#framegroup = frame.groupby(labelcolumn);
	# labelscount = Counter(frame[frame["TUACTIVITY_N"] == 1][labelcolumn]);
	# #jg is for joingroup
	# for jg in framegroup:
	
	daycount = np.zeros(2)
	data = np.zeros([actcount,288*2])
	sum = np.zeros(288*2)
	dayset = {1}

	
	
	for ind, row in frame.iterrows():
		
		fullcode = row['actind']#tri[int(row['TRCODE'])] #FIXME
		
		# day =  int(row['TUDIARYDAY']) - 1;
		# day =  (0 if int(row['TUDIARYDAY']) in weekdayset else 1);
		day = 0 #because we don't know
		dayset.add(day);
		daycount[day] += 1;
		
		stop = np.floor(row['end']/5.0);
		start = np.floor(row['start']/5.0)
		startind = int(day * 288 + start)
		stopind = int(day * 288 + stop)
	
		data[fullcode,startind:stopind] += 1;
	
		sum[startind:stopind] += 1;

	#print("normalizing...")
	for i in range(len(data)):
		data[i] /= sum;



	
	# print("plotting label "+ str(jg[0]) + ", c:"+str(labelscount[jg[0]]))
	# x = np.arange(1.0,(8.0+(4.0/24.0)),(8.-1.0)/(288*7+48))
	x = np.arange(0.0,2.0,1.0/288.0)
	
	additive = np.zeros(288*2)
	
	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.set_ylim([0.0,1.0])

	datasum = np.cumsum(data, axis=0);
	ax.fill_between(x,0,datasum[0],facecolor=colors[0],linewidth=0.1)
	
	for i in range(len(colors)-1):
		ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],linewidth=0.1)

	plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title( "Label " +str(label) )
	F = plt.gcf();
	F.set_size_inches(12,8)
	F.set_dpi(300);
	F.savefig(prefix +"label-" +str(label) + ".png");
	plt.clf();
	#plt.show()

	fig, ax = plt.subplots()

	cneg = -1
	cpos = 1;

	for i,gr in enumerate(frame.groupby("instance")):
		g,df = gr;
		#df = df.sort_values(["TUACTIVITY_N",])
		patches = [];

		#df['start'] = (df['TUCUMDUR24']-df['TUACTDUR24'])/60.0
		#df['length'] = df['TUACTDUR24']/60.0
		df['st'] = df['start']/60.0
		df['ln'] = df['length']/60.0
		df['color']= df['actind'].apply(lambda x: colors[int(x)])
		day = 0 # (0 if df['TUDIARYDAY'].iloc[0] in weekdayset else 1);

		x = df['st'].values;
		y = np.zeros_like(x)

		if(day): 
			y[:] = cneg
			cneg -=1
		else:
			y[:] = cpos
			cpos += 1
		
		c = df['color'].values;
		w = df['ln'].values;
		h = np.ones_like(w)
		
		for xi,yi,wi,hi,ci in zip(x,y,w,h,c):
			patches.append(mpatch.Rectangle((xi,yi),wi,hi,color=ci,))
		
		p = PatchCollection(patches,match_original=True)

		ax.add_collection(p)

		# cn+=1;
		# if(cn > 500): break;
		

	# fig.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	ax.set_xlim((0.,24.))
	ax.set_ylim((cneg,cpos))
	plt.title( "Label " +str(label) )
	F = plt.gcf();
	F.set_size_inches(12,8)
	F.set_dpi(300);
	F.savefig(prefix +"label-" + str(label) + "-seqs.png");	
	plt.clf()
	# plt.show()

	#if(savemats): h5out.close();

def propkernel(start,end,edgeblock):
	edgewidth = len(edgeblock);
	out = np.zeros((1440,));


	out[start:end] += 1.0;
	
	
	sedge = max(start-edgewidth,0);
	slen = min(start, edgewidth);
	eedge = min(1440,(end+edgewidth))
	elen = min((1440-end), edgewidth)

	out[sedge:start] += edgeblock[:slen][::-1];
	out[end:eedge] += edgeblock[:elen];

	return out;

def propkernhalfgauss(sidewidth):
	invals = np.linspace(0.0,3.0,sidewidth);
	out = halfnorm.pdf(invals);
	# print(out/out.max());
	return out/out.max();

def actprop(acttable, actmapping, sidewidth=20, limit=0.15):

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	edgeblock = propkernhalfgauss(sidewidth);

	for i,df in acttable.iterrows():
		out[df.actind] += propkernel(df.start,df.end,edgeblock);
	
	# for i in range(len(out)):
	# 	out[i] = out[i] / (out[i].max()+0.0001);
	# out = out/float(casecount);
	limitcount = limit*casecount
	out = np.clip(out,0.0,limitcount)/limitcount;

	return out;


def nextkernel(start,length,alpha=0.0):
	
	klength = length*2.0

	kern = skewnorm.pdf(np.linspace(-3.0,3.0,klength), alpha);
	out = np.zeros((1440,));

	
	
	sedge = max(0,start-length);
	eedge = min(1440,start+length);
	skern = max(0, length-start);
	ekern = min(klength, klength + 1440 - start)

	# print(start, length ,sedge, eedge, skern, ekern);

	out[sedge:eedge] = kern[skern:ekern]

	return out;

def actnext(acttable, actmapping, sidelen=0.5, limit=0.25):

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	for i,df in acttable.iterrows():
		out[df.actind] += nextkernel(df.start,np.ceil(df.length*sidelen),alpha=-3.0);

	limitcount = limit*casecount
	out = np.clip(out,0.0,limitcount)/limitcount;

	return out;


def compkernel(end,length,alpha=0.0):
	
	klength = length*2.0

	kern = skewnorm.pdf(np.linspace(-3.0,3.0,klength), alpha);
	out = np.zeros((1440,));

	
	
	sedge = max(0,end-length);
	eedge = min(1440,end+length);
	skern = max(0, length-end);
	ekern = min(klength, length + 1440 - end)

	# print(end, length ,sedge, eedge, skern, ekern);

	out[sedge:eedge] = kern[skern:ekern]

	return out;

def actcomplete(acttable, actmapping, sidelen=0.5, limit=0.25):

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	for i,df in acttable.iterrows():
		out[df.actind] += compkernel(df.end,np.ceil(df.length*sidelen),alpha=3.0);

	limitcount = limit*casecount
	out = np.clip(out,0.0,limitcount)/limitcount;

	return out;

def actaccum(acttable,actmapping, intervals=96 ,bins=96):

	

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount, intervals, bins));

	#group by activity, then cases
	groups = acttable.groupby(["actind","TUCASEID"])

	#populate histogram
	for g in groups:
		ind,fr = g;
		actind,case = ind;
		accum = np.zeros((1440,));

		for i,df in fr.iterrows():
			accum[df.start:df.end] += 1;

		accum = np.cumsum(accum)*(bins/1440.0)
		accum = accum[::int(np.floor(1440/intervals))]
		accum = accum.astype(int);
		out[actind][np.arange(intervals),accum] += 1;

	for i in range(len(out)):
		for j in range(len(out[i])):
			sumval = np.sum(out[i,j])
			if(sumval == 0): continue;
			out[i,j] /= sumval;#casecount; #np.amax(out[i,j]);

	#the beta fit is fraught with problems; 
	#it woudl make more sense to capture the kernel function instead

	x = np.linspace(0.0,1.0,bins)
	kernout = np.copy(out);

	for i in range(len(kernout)):
		for j in range(len(kernout[i])):
			rbfsmooth = Rbf(x,kernout[i,j],smooth=5.0)
			kernout[i,j] = rbfsmooth(x);
			ksum = np.sum(kernout[i,j])
			if(ksum > 0.0):	kernout[i,j] /= ksum;
		# 	#print(np.sum(betaout[i,j]));
		# 	sample = np.random.choice(x,p=kernout[i,j],size=100)
		# 	print(kernout[i,j],sample)
		# 	kernel = gaussian_kde(sample)
		# 	kernout[i,j] = kernel.pdf(x);
	

	return out,kernout;
	# return out;



def scorefunc(frame, propmat, accum,actmapping):

	actcount = len(actmapping);

	# intervals,bins = accum[0].shape

	block = np.zeros((actcount,1440))

	score = 0.0;
	for i,df in frame.iterrows():
		act = df.actind
		#prop
		score += np.sum(propmat[act][df.start:df.end])
		#accum
		block[act][df.start:df.end] += 1;


	# for i in range(len(block)):
	# 	s = np.cumsum(block[i])*(bins/1440.0)
	# 	s = s[::int(np.floor(1440/intervals))]
	# 	s = s.astype(int);
	# 	score += np.sum(accum[i][np.arange(intervals),s])


	return score


def verticalnorm(mat):

	mat = mat.T;
	for i in range(len(mat)):
		mat[i] /= np.sum(mat[i]);

	return mat.T

def picklen(x, lens, jprob):
	# print(x,jprob[x]);
	win = jprob[x].sample(n=1,weights=jprob[x]).index[0]	
	out = lens.iloc[win][['lmin','lmax','lavg','lstd','lhist']]
	return out;


#defines a matrix for the order of probability windows
#for non-overlapping windows the activity starts are trivial
#for overlapping windows, it gives the probability that the activity 
#is performed before the other window
#this probability can then be used for a sorting function.
def gcompare(x):
	# return 

	return 0.0

def getPrecedeMat(df,wins):
	
	wincount = len(wins);
	precede = np.zeros((wincount,wincount));
	#g = df.groupby(['instance','win'])
	#instcount = len(g);

	g = df.groupby(['wins'])

	# print("instcount: ",instcount)
	for i,iwin in wins.iterrows():
		for j,jwin in wins.iterrows():
			if(i==j): continue;
			#non-overlapping windows
			if( iwin['wmax'] < jwin['wmin'] ):
				precede[i,j] = 1.0
			#overlapping windows
			else:
				pcount = 0.0;
				#mat = g.get_group(i).append(g.get_group(j));
				mat = pd.merge(g.get_group(i)[['instance','start']],g.get_group(j)[['instance','start']],how='outer',on='instance')
				# print("MAT", i ,j)
				# print(mat);
				count = len(mat)
				mat['start_x'].fillna(mat['start_x'].dropna().sample(n=count,replace=True).reset_index(drop=True),inplace=True)
				mat['start_y'].fillna(mat['start_y'].dropna().sample(n=count,replace=True).reset_index(drop=True),inplace=True)
				# print(mat)
				pcount = mat.apply(lambda x: np.float(x.start_x < x.start_y),axis=1).sum()

				precede[i,j] = pcount / np.float(count);
			# print(i,j,precede[i,j],":",end=' ')


	return precede;


def precsort(actind,precede):
	actlen = len(actind);
	
	pmat = np.random.rand(actlen,actlen);
	omat = np.zeros((actlen,actlen));

	for i in range(actlen):
		for j in range(i+1,actlen):
			result = pmat[i,j] < precede[actind[i],actind[j]]
			omat[i,j] = (1.0 if result else 0.0)
			omat[j,i] = (0.0 if result else 1.0)
	
	return np.sum(omat, axis=0);


def buildseqv2(wins,lens,jointprob,precede,whereprob, dropind):
	#start,end,length, actind

	winlen = len(wins)

	actlist = wins[np.random.rand(winlen) < wins['prob'].values];
	# print(actlist);
	try:
		actlist[['lmin','lmax','lavg','lstd','lhist']] = actlist.index.to_series().apply(picklen, args=(lens,jointprob));
	except KeyError:
		print("There was a keyerror on this iteration: ")
		print(actlist, jointprob, lens);
		return None;




	#ctlist['length'] = actlist.apply(lambda x: x.lstd * np.random.randn() + x.lavg, axis=1)
	actlist['length'] = actlist['lhist'].apply(lambda x: np.random.choice(x[1],p=x[0])).fillna(1.0);
	actlist['length'] = actlist['length'].apply(lambda x: (1.0 if x < 1.0 else np.floor(x)));


	actind = np.array(actlist.index);
	actlist['precscore'] = precsort(actind,precede)
	actlist = actlist.sort_values(['precscore','wavg','wmin','length','wmax',]);


	#eliminate travel acts with precomputed tables
	actlist = actlist.drop(dropind);

	#assign locations and start criticality
	#-start criticality means that the start of the activity is non-movable
	#-if criticality is true then previous activities are pushed earlier
	#-if criticality is false then future acts are pushed later
	#this is performed in the order of activities
	#trips are assigned an arbitrary travel time of 15 minutes because
	#trip times cannot be known yet
	actlist['locp'] = actlist.index.to_series().apply(lambda x: whereprob[x].sample(n=1,weights=whereprob[x]).index[0]).fillna(-1);
	actlist['prevloc'] = actlist['locp'].shift(1).fillna(-1);
	actlist['loccrit'] = np.random.choice( (0,1),size=len(actlist) );


	actlist.drop(['wincount','winuniq','density','ref','prob','wmin','wmax','wavg','wstd','lmin','lmax','lavg','lstd','precscore','lweight','validwin','lhist'],axis=1,inplace=True)	

	return actlist;

def multiseqv2(wins,lens,jointprob,precede,whereprob, size=100):
	df = pd.DataFrame();
	# propp = verticalnorm(propp);
	# nextp = verticalnorm(nextp);
	# endp = verticalnorm(endp);

	for i in range(size):
		out = buildseqv2(wins,lens,jointprob,precede,whereprob);
		out["instance"]=i;
		df = pd.concat((df, out),axis=0);
	
	return df;

def buildseq(propp, nextp, endp):
	#start,end,length, actind
	st = []; en = []; ac = [];

	actcount,mincount = propp.shape;
	
	endchoice = np.random.rand(3,mincount);
	histvals = np.zeros((actcount,))

	maxcut = 0.97
	choosecut = 0.5
	lagfactor = 2.0
	act = np.random.choice(np.where(propp[:,0] > 0.99)[0]);
	lastst = 0;
	lastact = act;

	for m in range(0,1440):

		#eligible to continue?
		#eligible to quit?
		#eligible to switch?

		if( endp[act,m] > endchoice[0,m] or endp[act,m] > maxcut ):
			clist = np.where(propp[:,m] > choosecut)[0]
			recentlist = np.where(histvals > 0.0)[0]
			clist = [ b for b in clist if b not in recentlist ];
			p = propp[clist,m]
			p = p / np.sum(p);
			#print(clist,p);
			# print(m,clist);
			if( len(clist) > 0 ):
				act = np.random.choice(clist,p=p);
				#if we are just continuing an activity then there isn't a change
				if(lastact != act): 
					#prevent this activity from happening too frequently
					#by adding a limiter based on how long the last activity
					#was - e.g., we are not going back to sleep for a while
					#if we just woke up, but we might take a nap later
					histvals[lastact] = (m - lastst)*lagfactor;
					ac += [lastact]; st += [lastst]; en += [m]
					lastst=m;
					lastact = act;
		histvals -= 1.0 #always decrease
		histvals -= propp[:,m] #decrease faster if we are eligible
		#histvals[histvals < 0.0] = 0.0;

	ac += [lastact]; st += [lastst]; en += [1440]
	
	seq = pd.DataFrame({"start":st,"end":en,"actind":ac});
	seq["length"] = seq["end"] - seq["start"]
	# print(seq);
	return seq;

def multiseq(propp, nextp, endp, size=100):
	df = pd.DataFrame();
	# propp = verticalnorm(propp);
	# nextp = verticalnorm(nextp);
	# endp = verticalnorm(endp);

	for i in range(size):
		out = buildseq(propp,nextp,endp);
		out["instance"]=i;
		df = pd.concat((df, out),axis=0);
	
	return df;

	'''
        1 Respondent's home or yard
        2 Respondent's workplace
        3 Someone else's home
        4 Restaurant or bar
        5 Place of worship
        6 Grocery store
        7 Other store/mall
        8 School
        9 Outdoors away from home
        10 Library
        11 Other place
        12 Car, truck, or motorcycle (driver)
        13 Car, truck, or motorcycle (passenger)
        14 Walking
        15 Bus
        16 Subway/train
        17 Bicycle
        18 Boat/ferry
        19 Taxi/limousine service
        20 Airplane
        21 Other mode of transportation
        30 Bank
        31 Gym/health club
        32 Post Office
        89 Unspecified place
        99 Unspecified mode of transportation 

        [-1  1  9 14  3 11 13  6  7 12  4  2  5 31 15 16 30  8 10 89 32 18 21 19
     17 20 -3 99]
        undefine: -1, -3, 89, 99, 32,
        transportation: 15, 16, 17, 18, 19, 20, 21,
        recreation: 4, 5, 7, 10, 30, 31,
        grocery: 6
        residental: 1, 3,
        woorkplace: 2,
        outdoors: 9
        private transportation: 11, 12, 13, 14
        school: 8
	'''
    
def locaProb(df):
   
	data = {'where':[15, 16, 17,18,19, 20, 21, 4, 5, 7, 10, 30, 31, 6, 1, 3, 2, 9, -1, -3, 89, 99, 32, 11, 12, 13, 14, 8], 'change':[0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.93, 0.93,0.93, 0.93, 0.93, 0.93, 1.06, 1.08, 1.08, 0.62, 2.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
	changing = pd.DataFrame(data) 
	#print("localProb")
	#print(changing) 		whereprob = df.groupby(['wins']).apply(lambda x: x['where'].value_counts() / x['where'].count());

	print(df)    
	prob = df.apply(lambda x: x['where'].value_counts() / x['where'].count())
	print(prob)  

def assignChange(x, changing):
    #print(x)
    #print(x.values[0])
    #print(changing)
    out = changing.loc[(changing['where'] == x.values[0])]['change'].values[0]
    #print(out)
    return out
                       

###############
#    BEGIN    #
###############
def main():
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

	acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")
	#print(acttable)
	#print(acttable.columns)
	#acttable.info()

	labeltab = pd.read_csv(datapath + "final-label-classifier/labels.csv")
	# labeltab.info()

	acttable = pd.merge(acttable,labeltab,on="TUCASEID");

	acttable = acttable[['TUCASEID','TRCODE','daytypelabelreduce','TEWHERE','TUACTDUR24','TUCUMDUR24','TUACTIVITY_N']]
	#acttable.info()
	#print(acttable.groupby(['daytypelabelreduce']).size())    
	actmapping = np.sort(list(set(acttable['TRCODE'])))
	#print(actmapping)#actcount = len(mapping)
	ati = { tr:i for i,tr in enumerate(actmapping) }
	ita = { i:tr for i,tr in enumerate(actmapping) }
	# print([(i,k) for i,k in enumerate(actmapping)]);

	locmapping = np.sort(list(set(acttable['TEWHERE'])))
	#print(locmapping)    
	wti = { tr:i for i,tr in enumerate(locmapping) }
	itw = { i:tr for i,tr in enumerate(locmapping) }
	# print([(i,k) for i,k in enumerate(locmapping)]);

	labellist = np.sort(list(set(acttable['daytypelabelreduce'])))
	#print(labellist)
	#print(acttable)   
	#datapath = "/uufs/chpc.utah.edu/common/home/u1264235/test/output/"
	outData = "/uufs/chpc.utah.edu/common/home/u1264235/actMode/"
	outfile = h5py.File(outData + "actwindows.h5","w");
	outfile.create_dataset("/actmapping",data=actmapping,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.create_dataset("/locmapping",data=locmapping,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.create_dataset("/labels",data=labellist,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.close();

	acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
	acttable['end'] = acttable['TUCUMDUR24']
	acttable['length'] = acttable['TUACTDUR24']
	acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);
	acttable['instance'] = acttable['TUCASEID']
	acttable['where'] = acttable['TEWHERE'];
	#acttable.info()
	c = 0;
	samplecount = 400
	sample = np.random.choice(acttable.TUCASEID.unique(),size=samplecount,replace=False)
	#print(sample)
	#print(len(sample)) 
	randosample = acttable[acttable.TUCASEID.isin(sample)];

	# for i in range(0, len(actmapping)):
	# 	print("index:",i);
	# 	stenmixture(acttable,i,ncomp=15);
	# 'daytypelabelreduce','actind'

	#h5out = h5py.File(datapath + "actwindows.h5")

	#print(acttable['where'].unique())
    
	data = {'where':[15, 16, 17,18,19, 20, 21, 4, 5, 7, 10, 30, 31, 6, 1, 3, 2, 9, -1, -3, 89, 99, 32, 11, 12, 13, 14, 8], 'change':[0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.93, 0.93,0.93, 0.93, 0.93, 0.93, 1.06, 1.08, 1.08, 0.62, 2.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
	changing = pd.DataFrame(data) 
    
	for i,df in acttable.groupby(['daytypelabelreduce']):
		# if i not in [6,]: continue;
		print("calc:",i)
		# if(len(df) < 10): continue;
		#print(df)
		wins, lens = getwindows(df);

		#there's a weird interaction here where the window sorting is actually really important
		#the activity could be asasigned to more than one start window
		#but assignment only takes the first window
		#by sorting with descending start time, we guarantee that we always assign a window
		#preventing the issue with the joint probability window lacking an 
		#index for the window

		wins = wins.sort_values(['wmin','prob'],ascending=[False,False]);
		# print(wins,lens)
		# print(df.lwins.value_counts())
		df['wins'] = df[['actind','start']].apply(assignWindow,args=(wins,),axis=1);
		df['lwins'] = df[['actind','length']].apply(assignLen,args=(lens,),axis=1);
		df = df.sort_values(['instance','TUACTIVITY_N'])
		#print(df)
		#print(wins)
		#print(df.columns)        
		jointprob = df.groupby(['wins']).apply(lambda x: x['lwins'].value_counts() / x['lwins'].count() );
		print(jointprob);
		#print(df)
		#print(changing)        
		#df.merge(changing, left_on='where', right_on='where', how='left')  
		df["changing"] = df[['where']].apply(assignChange, args=(changing,), axis=1)        
		#print(df)  
		#whereprob = df.groupby(['wins', 'changing']).apply(lambda x: x['where'].value_counts() / x['where'].count());
		#print(whereprob) 
		whereprob = df.groupby(['wins', 'changing']).apply(lambda x: x['where'].value_counts() / x['where'].count() * x['changing'].values[0]);
		print(whereprob)  
        
        #changing.loc[(changing['where'] == x.values[0])]['change'].values[0]
		#whereprob = df.groupby(['wins']).apply(lambda x: (x['where'].value_counts() / x['where'].count()*x.loc[(x["where"]==)]));    
		#print(whereprob)        
		#whereprob = locaProb(df)
		#whereprob = df.groupby(['wins']).apply(locaProb)      
		#print(whereprob);

		precede = getPrecedeMat(df,wins);

		wins.to_hdf(outData + "actwindows.h5","/label-"+str(i)+"/windows",complib='zlib',complevel=9,mode='a');
		lens.to_hdf(outData + "actwindows.h5","/label-"+str(i)+"/lengthwin",complib='zlib',complevel=9,mode='a');
		jointprob.to_hdf(outData + "actwindows.h5","/label-"+str(i)+"/jointprob",complib='zlib',complevel=9,mode='a');
		whereprob.to_hdf(outData + "actwindows.h5","/label-"+str(i)+"/whereprob",complib='zlib',complevel=9,mode='a');
		pd.DataFrame(precede).to_hdf(outData + "actwindows.h5","/label-"+str(i)+"/precede",complib='zlib',complevel=9,mode='a');


		# h5out.create_dataset("/label-"+str(i)+"/windows",data=wins,compression='gzip',compression_opts=9);
		# h5out.create_dataset("/label-"+str(i)+"/lengthwin",data=lens,compression='gzip',compression_opts=9);
		# h5out.create_dataset("/label-"+str(i)+"/jointprob",data=jointprob,compression='gzip',compression_opts=9);
		# h5out.create_dataset("/label-"+str(i)+"/precede",data=precede,compression='gzip',compression_opts=9);

		#print("building seq...")
		#seqs = multiseqv2(wins,lens,jointprob,precede,size=200);

		#print("plotting...")
		#demoActPlot(seqs,i,actmapping);
		# break;
		
		# c += 1
	#h5out.close();

if __name__ == "__main__":
	main()