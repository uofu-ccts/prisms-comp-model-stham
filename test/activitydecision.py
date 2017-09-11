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
import numpy as np;
import sys;

pd.set_option('display.max_rows', 2000)


def sliceplot(mat):

	ran = np.max(mat[:,2])
	print(ran);
	cm_subsection = np.linspace(0.0,1.0, ran+1)
	maincolors = [ cm.jet(x) for x in cm_subsection ]
	for i,color in enumerate(maincolors):
		plt.scatter(mat[mat[:,2] == i,0],mat[mat[:,2] == i,1],color=color);
	plt.show()


def breaks1d(df, ncomp=10):
	mat = df.values;
	# print(matX)
	bgm = BayesianGaussianMixture(n_components=ncomp,covariance_type='full',max_iter=500,n_init=4).fit(mat);
	pred = bgm.predict(mat);

	return pred,bgm;


def assignWindow(x,wins):
	out = wins[wins['actind']==x['actind']][wins['wmax']>=x['start']][wins['wmin']<=x['start']].index;
	if(len(out) < 1): return -1;
	return out[0];

def assignLen(x,lens):
	out = lens[lens['actind']==x['actind']][lens['lmax']>=x['length']][lens['lmin']<=x['length']] .index;
	if(len(out) < 1): return -1;
	return out[0];

def getwindows(df, ncomp=10):
	ncases = df['instance'].unique().size
	# print(ncases,len(df));

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
		
		windows['ref'] = windows.index;
		windows['prob'] = windows['winuniq'] / ncases;
		windows['wmin'] = g.groupby('window').apply(lambda x: x['start'].min());
		windows['wmax'] = g.groupby('window').apply(lambda x: x['start'].max());
		windows['actind'] = i;

		allwindows = allwindows.append(windows);

		# print(g['window'].cov(g['lenwin']) )

		lengths =pd.DataFrame(index=g['lenwin'].unique());
		lengths['ref'] = lengths.index;
		lengths['lmin'] = g.groupby('lenwin').apply(lambda x: x['length'].min());
		lengths['lmax'] = g.groupby('lenwin').apply(lambda x: x['length'].max());
		lengths['actind'] = i;



		# print(lengths);
		alllengths = alllengths.append(lengths);

		#print( g.groupby('window').apply(lambda x: x['window'].cov(x['lenwin'])) )
		#print(i,g.groupby(['window']).apply(lambda x: x['lenwin'].value_counts() / x['lenwin'].count() ) );
		#print(i, g.groupby(['instance']).apply(lambda x: tuple(x['window'].sort_values().unique())).value_counts() )


		# print(windows);
	allwindows = allwindows.reset_index();
	allwindows = allwindows.drop("index",axis=1);
	alllengths = alllengths.reset_index();
	alllengths = alllengths.drop("index",axis=1);
	#print(allwindows,alllengths);
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
	print("Count:",count);

	mat = df[["start","length"]].values
	bgm = BayesianGaussianMixture(n_components=count,covariance_type='full',max_iter=500,n_init=10).fit(mat,matXYZ);
	matPred = bgm.predict(mat);
	
	# print(sti,matXYZ)
	print("Final:", len(list(set(matPred))))


	cm_subsection = np.linspace(0.0,1.0, count)
	maincolors = [ cm.jet(x) for x in cm_subsection ]
	for i,color in enumerate(maincolors):
		plt.scatter(mat[matPred==i,0],mat[matPred==i,1],color=color);
	plt.show();



	#return out;

def demoActPlot(frame,label,actmapping):

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
	# F = plt.gcf();
	# F.set_size_inches(12,8)
	# F.set_dpi(300);
	# F.savefig(prefix +"label-" +str(label) + ".png");
	# plt.clf();
	plt.show()

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
	# F = plt.gcf();
	# F.set_size_inches(12,8)
	# F.set_dpi(300);
	# F.savefig(prefix +"label-" +str(jg[0]) + "-seqs.png");	
	# plt.clf()
	plt.show()

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
	return lens.iloc[win][['lmin','lmax']]


def buildseqv2(wins,lens,jointprob):
	#start,end,length, actind
	st = []; en = []; ac = [];

	winlen = len(wins)

	actlist = wins[np.random.rand(winlen) < wins['prob'].values];
	# print(actlist);
	actlist[['lmin','lmax']] = actlist.index.to_series().apply(picklen, args=(lens,jointprob));

	actlist = actlist.sort_values(['wmin','lmax','wmax']);

	print(actlist);


	lastst = 0;
	lastact = 0;

	for m in range(0,1440):
		pass;

	ac += [lastact]; st += [lastst]; en += [1440]
	
	seq = pd.DataFrame({"start":st,"end":en,"actind":ac});
	seq["length"] = seq["end"] - seq["start"]
	# print(seq);
	return seq;

def multiseqv2(wins,lens,jointprob, size=100):
	df = pd.DataFrame();
	# propp = verticalnorm(propp);
	# nextp = verticalnorm(nextp);
	# endp = verticalnorm(endp);

	for i in range(size):
		out = buildseqv2(wins,lens,jointprob);
		out["instance"]=i;
		df = pd.concat((df, out),axis=0);
	
	return df;

def buildseq(propp, nextp, endp):
	#start,end,length, actind
	st = []; en = []; ac = [];

	actcount,mincount = propp.shape;
	
	endchoice = np.random.rand(3,mincount);
	histvals = np.zeros((actcount,))
	# a = propp > choice[0]; #continue
	# b = nextp > choice[1]; #switch
	# c = endp > choice[2]; #quit

	# print(propp[:,0])
	# print(a[:,0])
	# print(np.where(a[:,0]))
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

###############
#    BEGIN    #
###############

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")

labeltab = pd.read_csv(datapath + "newclassify-final/labels.csv")

acttable = pd.merge(acttable,labeltab,on="TUCASEID");

acttable = acttable[['TUCASEID','TRCODE','daytypelabelreduce','TEWHERE','TUACTDUR24','TUCUMDUR24','TUACTIVITY_N']]

actmapping = np.sort(list(set(acttable['TRCODE'])))
#actcount = len(mapping)
ati = { tr:i for i,tr in enumerate(actmapping) }
ita = { i:tr for i,tr in enumerate(actmapping) }
print([(i,k) for i,k in enumerate(actmapping)]);

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
# print([(i,k) for i,k in enumerate(locmapping)]);

labellist = np.sort(list(set(acttable['daytypelabelreduce'])))

acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
acttable['end'] = acttable['TUCUMDUR24']
acttable['length'] = acttable['TUACTDUR24']
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);
acttable['instance'] = acttable['TUCASEID']

c = 0;
samplecount = 400
sample = np.random.choice(acttable.TUCASEID.unique(),size=samplecount,replace=False)
randosample = acttable[acttable.TUCASEID.isin(sample)];

# for i in range(0, len(actmapping)):
# 	print("index:",i);
# 	stenmixture(acttable,i,ncomp=15);
# 'daytypelabelreduce','actind'
for i,df in acttable.groupby(['daytypelabelreduce']):
	if i not in [6,]: continue;
	print("calc:",i)
	# if(len(df) < 10): continue;
	
	wins, lens = getwindows(df);

	#there's a weird interaction here where the window sorting is actually really important
	#the activity could be asasigned to more than one start window
	#but assignment only takes the first window
	#by sorting with descending start time, we guarantee that we always assign a window
	#preventing the issue with the joint probability window lacking an 
	#index for the window

	wins = wins.sort_values(['wmin','prob'],ascending=[False,False]);
	print(wins,lens)
	# print(df.lwins.value_counts())
	df['wins'] = df[['actind','start']].apply(assignWindow,args=(wins,),axis=1);
	df['lwins'] = df[['actind','length']].apply(assignLen,args=(lens,),axis=1);

	

	jointprob = df.groupby(['wins']).apply(lambda x: x['lwins'].value_counts() / x['lwins'].count() );
	print(jointprob);
	# print( df.groupby('wins').apply(lambda x: x['wins'].cov(x['lwins'])) )
	#buildseqv2(wins,lens,jointprob);
	multiseqv2(wins,lens,jointprob,size=5);
	# print(df.groupby(['instance']).apply(lambda x: tuple(x['wins'].sort_values().unique())).value_counts() )
	#print(df.iloc[0:10]);
	# stenmixture(df,i,ncomp=10);
	break;
# 	if i not in (1,6,9): continue;

# 	if (c >= 10): break;
# 	print("calc:",i)


# 	ind = 189;

# 	stenmixture(df, ind,ncomp=15);

	
	#actcount,1440
	# prop = actprop(df,actmapping);
	# plt.matshow(prop);
	# plt.show();

	# limit = df[df['actind'] == ind]
	# limit["length"] = 5;
	# demoActPlot(limit,0,actmapping);

	#actcount,1440
	#nextp = actnext(df,actmapping,sidelen=0.5,limit=0.15);
	#actcount,1440
	#comp = actcomplete(df,actmapping,sidelen=0.5,limit=0.15);
	#actcount,intervals,bins
	#hist,kernhist = actaccum(df,actmapping,intervals=96,bins=96);



	#print("building seq...")
	#seq = buildseq(prop, nextp, comp)

	#seqs = multiseq(prop, nextp, comp)
	#print("plotting...")
	#demoActPlot(seqs,i,actmapping);
	#print(seq);


	# print("scoring");
	# casemap = np.sort(list(set(df['TUCASEID'])))
	# casecount = len(casemap)
	# scores = np.zeros((casecount,));
	# altscores = np.zeros((samplecount));

	# n = 0
	# for j,dfj in df.groupby(["TUCASEID"]):
	# 	scores[n] = scorefunc(dfj,prop,kernhist,actmapping);
	# 	n+=1

	# n = 0
	# for j,dfj in randosample.groupby(["TUCASEID"]):
	# 	altscores[n] = scorefunc(dfj,prop,kernhist,actmapping);
	# 	n+=1

	# bins = 100;

	# scorehist = np.histogram(scores,bins=bins,normed=True,range=(0.0,np.amax(scores)))

	# altscorehist = np.histogram(altscores,bins=bins,normed=True,range=(0.0,np.amax(altscores)))

	# print(scorehist,altscorehist);

	# plt.plot(scorehist[1][:-1],scorehist[0]);
	# plt.plot(altscorehist[1][:-1],altscorehist[0]);
	# plt.title(str(i))
	# #106
	# # plt.matshow(hist[106]);
	# # plt.title("hist")
	
	# # plt.matshow(kernhist[106]);
	# # plt.title("kernhist")
	# plt.show();
	
	
	c += 1