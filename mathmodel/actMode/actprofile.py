import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
from matplotlib.collections import PatchCollection;
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

pd.set_option('display.max_rows', 2000)
np.set_printoptions(threshold=np.inf)
###############################################
#   DEFINES
###############################################
def dbscangrid(coords,prefix, eps=[0.005,0.01,0.02,0.03,0.04,0.05,0.1],sample =[1,2,3,5,10,20,50,100]):
	coordsT = coords.T;
	_eps = eps
	_sample = sample
	for i in eps:
		for j in _sample:
			dbscan = sklearn.cluster.DBSCAN(eps=i, min_samples=j);
			dblabels = dbscan.fit_predict(coords);
			compPlot(coordsT,dblabels,prefix+"-"+str(i)+"-"+str(j));

def featureSort(clf,names):
	feats = clf.feature_importances_
	df = pd.DataFrame({'feat':feats,'names':names});
	df = df.sort_values('feat',ascending=False)
	return df;
def demoActPlot(frame,labelcolumn, prefix,savemats=False,cutoff=25):

	if(savemats):
		h5out = h5py.File(prefix+"mats.h5")

		

	weekdayset = [2,3,4,5,6]
	#prelim stuff
	mapping = np.sort(list(set(frame['TRCODE'])))
	actcount = len(mapping)
	tri = { tr:i for i,tr in enumerate(mapping) }

	
	#set up coloration
	start = 0.0
	stop = 1.0
	number_of_lines=19
	cm_subsection = np.linspace(start, stop, number_of_lines)
	maincolors = [ cm.prism(x) for x in cm_subsection ]
	colors = []
	for i in mapping:
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

	framegroup = frame.groupby(labelcolumn);
	labelscount = Counter(frame[frame["TUACTIVITY_N"] == 1][labelcolumn]);
	#jg is for joingroup
	for jg in framegroup:
		
		if(labelscount[jg[0]] < cutoff):
			continue;
		
		daycount = np.zeros(2)
		data = np.zeros([actcount,288*2])
		sum = np.zeros(288*2)
		dayset = {1}
		
		# #print(jg[1])
		# subsetjg = jg[1][jg[1]['TUACTIVITY_N'] == 1]
		# mdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[1]
		# fdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[2]
		# #print(mdem,fdem);
		# if(type(mdem) == pd.core.series.Series): mdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='b')
		# else: print("bad mdem:",mdem,type(mdem))
		# if(type(fdem) == pd.core.series.Series): fdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='r')
		# else: print("bad fdem:",fdem,type(fdem))

		# plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		# F = plt.gcf();
		# F.set_size_inches(4,3)
		# F.set_dpi(300);
		# F.savefig(prefix + "demo-" +str(jg[0]) + ".png");
		# plt.clf();
		
		
		for ind, row in jg[1].iterrows():
			
			fullcode = tri[int(row['TRCODE'])]
			
			day =  int(row['TUDIARYDAY']) - 1;
			day =  (0 if int(row['TUDIARYDAY']) in weekdayset else 1);
			dayset.add(day);
			daycount[day] += 1;
			
			stop = np.floor(row['TUCUMDUR24']/5.0);
			start = stop - np.floor(row['TUACTDUR24']/5.0)
			startind = int(day * 288 + start)
			stopind = int(day * 288 + stop)
		
			data[fullcode,startind:stopind] += 1;
		
			sum[startind:stopind] += 1;

		#print("normalizing...")
		for i in range(len(data)):
			data[i] /= sum;

		if(savemats):
			ds = h5out.create_dataset('/label-'+str(jg[0]),data=data,fillvalue=0.,compression='gzip',compression_opts=9);
			ds.attrs['label']=jg[0];
			ds.attrs['prefix']=prefix;
		



		
		print("plotting label "+ str(jg[0]) + ", c:"+str(labelscount[jg[0]]))
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
		plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		F = plt.gcf();
		F.set_size_inches(12,8)
		F.set_dpi(300);
		F.savefig(prefix +"label-" +str(jg[0]) + ".png");
		plt.clf();

	# def plotseq(frame):
		fig, ax = plt.subplots()

		cneg = -1
		cpos = 1;

		for i,gr in enumerate(jg[1].groupby("TUCASEID")):
			g,df = gr;
			df = df.sort_values(["TUACTIVITY_N",])
			patches = [];

			df['start'] = (df['TUCUMDUR24']-df['TUACTDUR24'])/60.0
			df['length'] = df['TUACTDUR24']/60.0
			df['color']= df['TRCODE'].apply(lambda x: colors[tri[int(x)]])
			day =  (0 if df['TUDIARYDAY'].iloc[0] in weekdayset else 1);

			x = df['start'].values;
			y = np.zeros_like(x)

			if(day): 
				y[:] = cneg
				cneg -=1
			else:
				y[:] = cpos
				cpos += 1
			
			c = df['color'].values;
			w = df['length'].values;
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
		plt.title( "Label " +str(jg[0]) )
		F = plt.gcf();
		F.set_size_inches(12,8)
		F.set_dpi(300);
		F.savefig(prefix +"label-" +str(jg[0]) + "-seqs.png");	
		plt.clf()

	if(savemats): h5out.close();



def treeRefit(vector, labels,depth=None,nest = 500):
	new_maxleaf=None;
	new_sample_size = 1
	new_split_size = 2
	#steps = nest // 100
	#new_n_est = nest // steps
	new_n_est = nest
	
	new_maxdepth = depth

	clf = sklearn.ensemble.ExtraTreesClassifier(n_jobs=1,max_leaf_nodes=new_maxleaf,n_estimators=new_n_est,criterion='entropy',min_samples_split=new_split_size,min_samples_leaf=new_sample_size,max_depth=new_maxdepth);
	
	clf = clf.fit(vector,labels);

	#warm start fix for solving the out of control mem issue ???
	#for n in range(steps):
	#	clf = clf.fit(vector,labels);
	#	clf.n_estimators += new_n_est;
	

	newlabels = clf.predict(vector);
	print(newlabels.shape)
	return clf, newlabels;

def randoTreeRefit(vector, labels,depth=None,nest = 500):
	new_maxleaf=None;
	new_sample_size = 1
	new_split_size = 2
	new_n_est = nest
	
	new_maxdepth = depth

	clf = sklearn.ensemble.RandomTreesEmbedding(n_jobs=1,max_leaf_nodes=new_maxleaf,n_estimators=new_n_est,min_samples_split=new_split_size,min_samples_leaf=new_sample_size,max_depth=new_maxdepth);
	
	clf = clf.fit(vector,labels);	

	newlabels = clf.predict(vector);
	return clf, newlabels;

def randoTrees(vector, depth=5, nest = 500):
	split_size = 2
	sample_size = 1
	maximpure = 0.00001
	maxleaf = None
	n_est = nest
	maxdepth = depth

	clf = sklearn.ensemble.RandomTreesEmbedding(max_leaf_nodes=maxleaf,n_estimators=n_est,min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,min_impurity_decrease=maximpure)
	clf = clf.fit(vector);
	labels = clf.apply(vector);
	return clf, labels;

def tsneLabelFit(prox, eps = 0.05, samples = 10,perplex=10):
	ncomp = 2;
	print(prox.shape)
	tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=perplex,early_exaggeration=10.0,verbose=2,metric='precomputed', n_iter=1000,learning_rate=600)
	coords = tsne.fit_transform(1-prox)
	print(coords.shape)
	coords /= np.max(coords);
	dbscan = sklearn.cluster.DBSCAN(eps = eps, min_samples= samples);
	#dblabels = dbscan.fit_predict(prox);
	dblabels = dbscan.fit_predict(coords);
	return coords,dblabels;


def compPlot(coords, labels, prefix):
	neglabels = labels #= -1 # == or = ?
	labels = labelReduce(labels);
	labels[neglabels] = -1;
	ncomp = len(coords);
	cmapp = np.linspace(0.0,1.0,len(set(labels)));
	colors = [ cm.jet(x) for x in cmapp ]
	#np.random.shuffle(colors);
	outc = [ colors[b] if b > -1 else (0,0,0,1) for b in labels ];
	for i in range(ncomp):
		for j in range(i+1,ncomp):
			plt.scatter(coords[i],coords[j],color=outc,s=8,edgecolor='');
			plt.title("coords:"+","+str(i)+","+str(j))
			# plt.show()
			F = plt.gcf();
			F.set_size_inches(8,6)
			F.set_dpi(300);
			F.savefig(prefix +"-" +str(i)+"-"+str(j)+ ".png");
			plt.clf();

def writeTree(name, clf, mapping):
	dotdata = sklearn.tree.export_graphviz(clf, feature_names=[str(b) for b in mapping],out_file=None, rotate=True,class_names=True,node_ids=True);
	dotdata = re.sub(r"value = \[([^]]+)\]", "", dotdata);
	#print(dotdata)
	g = pydotplus.graph_from_dot_data(dotdata);
	g.write_png(name);


def outerequal(x):
	tcount = len(x[0]);
	print(tcount)
	print(x[0])
	prox = np.zeros( (tcount,tcount) );
	for i in range(len(x)):
        
		prox += np.equal.outer(x[i],x[i]);
	print(prox)
	return prox;

def proxMat(labels,procs=4):
	print("Calculating proximity matrix...")
	print(len(labels))
	tcount = len(labels);
	n_est = len(labels[0]);

	lines = labels.T;
	prox = np.zeros( (tcount,tcount) );
    
	llen = len(lines)
	tick = llen // 100;
	for i in range(len(lines)):
		if(i%tick==0): print(i//tick,end="..",flush=True);
       
		prox += np.equal.outer(lines[i],lines[i]);
	#print(np.equal.outer(lines[-1],lines[-1]).shape)
	#print(np.equal.outer(lines[-1],lines[-1]))
	#print(prox[-1])
	print(prox.shape)
	'''
	proccount = procs

	pool = Pool(proccount);

	lines = labels.T;

	#llen = len(lines)
	#tick = llen // 100;
	#for i in range(len(lines)):
	#	if(i%tick==0): print(i//tick,end="..",flush=True);
	#	prox += np.equal.outer(lines[i],lines[i]);

	split = np.array_split(lines,proccount);

	result = pool.map(outerequal, split);
	print(result.shape)
	for i in range(len(result)):
		prox += result[i];

	pool.close();
	'''
    
	#print("  ")
	prox = prox / float(n_est);

	print("Proximity percentiles...")
	print([0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99])
	print(np.percentile(prox,[0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99]))
	return prox

def labelReduce(labels):
	labelset = set(labels);
	labelmap = { item:index for index,item in enumerate(labelset) }
	newlabels = np.array([labelmap[b] for b in labels]);
	return newlabels;

def tuccconv(x):
	if(x == '-1'): return -1;
	if(x == '-2'): return -2;
	if(x == '-3'): return -3;
	b = datetime.strptime(x,'%H:%M:%S')
	return b.hour*60 + b.minute;

def columnStrip(frame, columns):
	framecolumns = [col for col in frame.columns if col in columns];
	subframe = frame[framecolumns].copy();
	return subframe;

def labelCutoff(frame, labels, cutoff):
	counts = Counter(labels);
	relabelfunc = np.vectorize(lambda x: x if counts[x] >= cutoff else -1);
	relabel = relabelfunc(labels);
	relabelframe = frame[relabel > -1];
	relabel = relabel[relabel > -1];
	return relabelframe, relabel;

def stripUnlabelled(frame,labels,badlabel):
	stripframe = frame[labels > -1]
	striplabel = labels[labels > -1]
	return stripframe, striplabel;

def determineLabels(frame, columns, name, path, cutoff = 50, eps=0.3, samples=10,perplex=10,savefinalclf=False):
	#per http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#workings,
	#there is a means to use a two class method to estimate effectiveness
	#instead we can just use the proximity matrix method with random trees
	#without the splits, especially where there are a high number of components
	#TSNE works better than MDS in this case, but YMMV. 
	labelframe = pd.DataFrame(frame['TUCASEID']);
	print("Processing :", name, " with columns: ", columns);
	subframe = columnStrip(frame, columns);
	vector = subframe.values;
	print(vector.shape)
	print("Initial random forest fitting...")
	initclf, initlabels = randoTrees(vector,depth=5,nest=2000);
	print(initlabels.shape)
	#print(initlabels[:2])    
    
	prox = proxMat(initlabels,procs=4);
	coords, dblabels = tsneLabelFit(prox, eps=eps, samples = samples,perplex=perplex);
	coordsT = coords.T;

	dbscangrid(coords,path+"/"+name+"-dbscan-grid")

	print("Init label count: ",len(Counter(dblabels))-1)
	compPlot(coordsT,dblabels,path+"/"+name+"-dbscan-init");
	labelframe[name+'labelinit'] = dblabels;

	print("Random forest re-fitting...")
	stripvector, striplabels = stripUnlabelled(subframe, dblabels, -1)
	print("Stripped size: ", len(striplabels));
	refitclf,refitlabels = treeRefit(stripvector, striplabels, depth=8, nest=2000);
	refitlabels = refitclf.predict(vector);

	print("Refit label count: ",len(Counter(refitlabels)))
	compPlot(coordsT,refitlabels,path+"/"+name+"-dbscan-refit");
	labelframe[name+'labelrefit'] = refitlabels;

	print("Relabeling sets with size <", cutoff, "...")
	#### POSSIBLE MAJOR BUG
	#### this section of code may mix up columns badly
	#### alternate explanation is that joblib changed with anaconda
	#### messing up the order of columsnb of values, or some such
	reducedframe, reducedlabels = labelCutoff(subframe, refitlabels, cutoff);
	relabelclf,reducedlabels=treeRefit(reducedframe.values,reducedlabels,depth=None, nest=2000);
	reducedlabels = relabelclf.predict(vector);

	print("Reduced label count: ",len(Counter(reducedlabels)))
	compPlot(coordsT,reducedlabels,path+"/"+name+"-dbscan-reduce");
	labelframe[name+'labelreduce'] = reducedlabels;

	if(savefinalclf):
		joblib.dump(relabelclf, path+"/clfsave-"+name+".pkl", compress=('gzip',9));

	featimport = featureSort(relabelclf, subframe.columns);

	return labelframe, featimport;



###############################################
#   BEGIN MAIN CODE
###############################################
def main():

	#tiercode = 'TRTIER2'
	tiercode = 'TRCODE'

	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"
	outpath = "/uufs/chpc.utah.edu/common/home/u1264235/test/output/"

	print("reading...")
	acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
	acttable.info()    
	infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
	infotable.info()
	rosttable = pd.read_csv(datapath + "atusrost_2015/atusrost_2015.dat")
	rosttable.info()
	#print(acttable.columns)
	#print(infotable.columns)
	#print(rosttable.columns)
	print("fixing roster...");
	rosttable = rosttable[rosttable['TERRP'].apply(lambda x: x in [18,19])]
	rosttable = rosttable.drop(['TXAGE','TXRRP','TXSEX','TULINENO','TERRP'],axis=1)

	print("joining, cleaning, mapping...")
	infotable = pd.merge(infotable,rosttable,on='TUCASEID')
	acttable = pd.merge(acttable,infotable[['TUCASEID','TUDIARYDAY','TESEX','TEAGE']],on='TUCASEID')

	# What is this? TUCC has datetimes embedded, need to translate
	# #force some cleanliness:
	infotable['TUCC2'] = infotable['TUCC2'].apply(tuccconv);
	infotable['TUCC4'] = infotable['TUCC4'].apply(tuccconv);

	mapping = np.sort(list(set(acttable[tiercode])))
	#print([(i,k) for i,k in enumerate(mapping)]);
	actcount = len(mapping)

	tri = { tr:i for i,tr in enumerate(mapping) }
	itr = { i:tr for i,tr in enumerate(mapping) }

	#daypick = [2,3,4,5,6]
	#daypick = [1,1]
	#daypick = [7,7]
	#daypick = [2,2]
	daypick = [1,2,3,4,5,6,7]

	acttable['mapped'] = acttable[tiercode].apply(lambda x: tri[x]);
	acttable = acttable[acttable['TUDIARYDAY'].apply(lambda x: x in daypick) ];
	infotable = infotable[infotable['TUDIARYDAY'].apply(lambda x: x in daypick) ];

	#acttable.info();

	print('Processing...')
	cases = acttable.groupby(['TUCASEID']) 
	print(cases)    
	casecount = len(cases);

	print("Casecount: ",casecount)
	#vectors = np.zeros((casecount,actcount))

	min_step = 5
	actsteps = 1440 // min_step

	acttable['start'] = (acttable['TUCUMDUR24']-acttable['TUACTDUR24'])//min_step
	acttable['end'] = acttable['TUCUMDUR24']//min_step
	acttable['length'] = acttable['end']-acttable['start']
	acttable['length'] = acttable['length'].apply(lambda x:1 if x <= 0 else x)

	vectors = np.zeros((casecount,actcount))
	vectors1 = np.zeros((casecount,actsteps))
	vectorsWh = np.zeros((casecount,actsteps))
	vectors2 = np.zeros_like(infotable.values);


	for ind,i in enumerate(cases):
		for n,j in i[1].iterrows():
			# print(j)
			vectors[ind][j['mapped']] += j['TUACTDUR24'];
		#print(i)
		g,df = i;
		df = df.sort_values(["TUACTIVITY_N"])
		vf = np.zeros((actsteps));
		vw = np.zeros((actsteps));
		for row in df.iterrows():
			# print(row[1]['start'],row[1]['length'],end=' ')
			vf[(row[1]['start']):(row[1]['start']+row[1]['length'])] = row[1][tiercode];
			vw[(row[1]['start']):(row[1]['start']+row[1]['length'])] = row[1]['TEWHERE'];
		
		# print(vf);
		vectors1[ind] = vf;
		vectorsWh[ind] = vw;
		vectors2[ind] = infotable[infotable['TUCASEID']==i[1]['TUCASEID'].iloc[0]].values;

	
	#supercolumns = [str(b) for b in mapping] + list(infotable.columns) ;
	minlist = [("min"+ str(i).zfill(4)) for i in range(0,1440,min_step)]
	wherelist = [("whr"+ str(i).zfill(4)) for i in range(0,1440,min_step)]
	supercolumns = [str(b) for b in mapping] + minlist + wherelist + list(infotable.columns) ;
	superframe = pd.DataFrame(np.concatenate((vectors,vectors1,vectorsWh,vectors2),axis=1),columns=supercolumns);
	superframe = superframe.set_index(infotable.index);
	superframe.info()    




	#print("Supercolumns: ",supercolumns);


	imgpath = outpath + "classify-" + time.strftime("%Y-%m-%d_%H-%M-%S")
	os.mkdir(imgpath)


	goodcols = ['TEAGE', 'TEHRUSL1', 'TELFS', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESEX', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRHHCHILD', 'TRSPPRES', 'TUDIS2', 'TUELNUM', 'TUSPUSFT']

	

	# fullcols = goodcols + ['TUDIARYDAY',
	#actcolumns = ['10101', '10102', '10199', '10201', '10299', '10301', '10399', '10401', '20101', '20102', '20103', '20104', '20199', '20201', '20202', '20203', '20299', '20301', '20302', '20303', '20399', '20401', '20402', '20499', '20501', '20502', '20601', '20602', '20699', '20701', '20799', '20801', '20899', '20901', '20902', '20903', '20904', '20905', '20999', '29999', '30101', '30102', '30103', '30104', '30105', '30106', '30108', '30109', '30110', '30111', '30112', '30199', '30201', '30202', '30203', '30204', '30299', '30301', '30302', '30303', '30399', '30401', '30402', '30403', '30404', '30405', '30499', '30501', '30502', '30503', '30504', '30599', '40101', '40102', '40103', '40104', '40105', '40106', '40108', '40109', '40110', '40111', '40112', '40199', '40201', '40202', '40301', '40302', '40303', '40399', '40401', '40402', '40403', '40404', '40405', '40499', '40501', '40502', '40503', '40504', '40505', '40506', '40507', '40508', '40599', '49999', '50101', '50102', '50103', '50104', '50199', '50201', '50202', '50203', '50299', '50301', '50302', '50303', '50304', '50399', '50401', '50403', '50404', '50499', '59999', '60101', '60102', '60103', '60104', '60199', '60201', '60202', '60203', '60204', '60299', '60301', '60302', '60399', '60401', '60402', '60499', '69999', '70101', '70102', '70103', '70104', '70105', '70201', '80101', '80102', '80201', '80202', '80203', '80301', '80401', '80402', '80403', '80499', '80501', '80502', '80601', '80701', '80702', '89999', '90101', '90102', '90103', '90104', '90199', '90201', '90202', '90299', '90301', '90302', '90399', '90401', '90501', '90502', '90599', '99999', '100101', '100102', '100103', '100199', '100201', '100299', '100304', '100305', '100401', '110101', '110201', '120101', '120201', '120202', '120299', '120301', '120302', '120303', '120304', '120305', '120306', '120307', '120308', '120309', '120310', '120311', '120312', '120313', '120399', '120401', '120402', '120403', '120404', '120405', '120499', '120501', '120502', '120503', '120504', '130101', '130102', '130103', '130104', '130105', '130106', '130107', '130108', '130109', '130110', '130112', '130113', '130114', '130115', '130116', '130117', '130118', '130119', '130120', '130122', '130124', '130125', '130126', '130127', '130128', '130129', '130130', '130131', '130132', '130133', '130134', '130135', '130136', '130199', '130202', '130203', '130207', '130210', '130212', '130213', '130216', '130218', '130219', '130220', '130222', '130223', '130224', '130225', '130226', '130227', '130229', '130232', '130299', '130301', '130302', '139999', '140101', '140102', '140103', '140105', '149999', '150101', '150102', '150103', '150104', '150105', '150106', '150199', '150201', '150202', '150203', '150204', '150299', '150301', '150302', '150399', '150401', '150402', '150501', '150601', '150602', '150699', '150701', '150801', '159999', '160101', '160102', '160103', '160104', '160105', '160106', '160107', '160108', '160199', '160201', '180101', '180201', '180202', '180203', '180204', '180205', '180206', '180207', '180208', '180209', '180299', '180301', '180302', '180303', '180304', '180305', '180401', '180402', '180403', '180404', '180405', '180499', '180501', '180502', '180503', '180504', '180599', '180601', '180602', '180603', '180604', '180699', '180701', '180702', '180703', '180704', '180801', '180802', '180803', '180804', '180805', '180806', '180807', '180899', '180901', '180902', '180903', '180904', '180905', '180999', '181001', '181002', '181101', '181201', '181202', '181203', '181204', '181205', '181299', '181301', '181302', '181399', '181401', '181501', '181599', '181601', '181801', '189999', '500101', '500103', '500105', '500106', '500107']
	actcolumns = [str(b) for b in mapping]

	fullcols = goodcols + ['TUDIARYDAY'] + minlist + actcolumns;
	#fullcols = ['TUDIARYDAY'] + minlist + actcolumns;
	# fullcols = ['TUDIARYDAY'] + minlist + wherelist + actcolumns;

	casetype,firstimport = determineLabels(superframe, goodcols, "casetype", imgpath, cutoff = 25, eps=0.03,samples=20,perplex=30,savefinalclf=True);

	daytype,secondimport = determineLabels(superframe, fullcols, "daytype", imgpath, cutoff = 25,eps=0.02,samples=10,perplex=30);


	casecol = pd.merge(casetype,daytype,on='TUCASEID')
	#casecol = daytype;
	outcol = pd.merge(casecol, superframe[['TUDIARYDAY','TUCASEID']],on='TUCASEID')

	print(firstimport)
	print(secondimport)
	#print(featureSort(finalclf,subframe.columns));
	#print("Percent same labels ", np.sum(casecol['finallabels']==casecol['secondlabelreduce'])/casecount*100)

	outcol.to_csv(imgpath+"/labels.csv")


	print("secondary processing...")
	labelledacttable = pd.merge(acttable,casecol,on='TUCASEID')

	#clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2,min_samples_leaf=5,max_depth=8)
	#subframe = columnStrip(superframe,goodcols)
	#clf = clf.fit(subframe.values,casecol['firstlabelreduce'].values);
	#writeTree(imgpath + "/dtreefinal.png",clf,subframe.columns)

	demoActPlot(labelledacttable, "casetypelabelreduce", imgpath + "/casetype-")
	demoActPlot(labelledacttable, "daytypelabelreduce", imgpath + "/daytype-",savemats=True)




	##############################################################################
	#supercolumns = ['TUCASEID', 'TULINENO', 'TUYEAR', 'TUMONTH', 'TEABSRSN', 'TEERN', 'TEERNH1O', 'TEERNH2', 'TEERNHRO', 'TEERNHRY', 'TEERNPER', 'TEERNRT', 'TEERNUOT', 'TEERNWKP', 'TEHRFTPT', 'TEHRUSL1', 'TEHRUSL2', 'TEHRUSLT', 'TEIO1COW', 'TEIO1ICD', 'TEIO1OCD', 'TELAYAVL', 'TELAYLK', 'TELFS', 'TELKAVL', 'TELKM1', 'TEMJOT', 'TERET1', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRDTIND1', 'TRDTOCC1', 'TRERNHLY', 'TRERNUPD', 'TRERNWA', 'TRHERNAL', 'TRHHCHILD', 'TRHOLIDAY', 'TRIMIND1', 'TRMJIND1', 'TRMJOCC1', 'TRMJOCGR', 'TRNHHCHILD', 'TRNUMHOU', 'TROHHCHILD', 'TRSPFTPT', 'TRSPPRES', 'TRTALONE', 'TRTALONE_WK', 'TRTCC', 'TRTCCC', 'TRTCCC_WK', 'TRTCCTOT', 'TRTCHILD', 'TRTCOC', 'TRTEC', 'TRTFAMILY', 'TRTFRIEND', 'TRTHH', 'TRTHHFAMILY', 'TRTNOCHILD', 'TRTNOHH', 'TRTO', 'TRTOHH', 'TRTOHHCHILD', 'TRTONHH', 'TRTONHHCHILD', 'TRTSPONLY', 'TRTSPOUSE', 'TRTUNMPART', 'TREMODR', 'TRWERNAL', 'TRYHHCHILD', 'TTHR', 'TTOT', 'TTWK', 'TUABSOT', 'TUBUS', 'TUBUS1', 'TUBUS2OT', 'TUBUSL1', 'TUBUSL2', 'TUBUSL3', 'TUBUSL4', 'TUCC2', 'TUCC4', 'TUCC5B_CK', 'TUCC5_CK', 'TUCC9', 'TUDIARYDATE', 'TUDIARYDAY', 'TUDIS', 'TUDIS1', 'TUDIS2', 'TUECYTD', 'TUELDER', 'TUELFREQ', 'TUELNUM', 'TUERN2', 'TUERNH1C', 'TUFINLWGT', 'TUFWK', 'TUIO1MFG', 'TUIODP1', 'TUIODP2', 'TUIODP3', 'TULAY', 'TULAY6M', 'TULAYAVR', 'TULAYDT', 'TULK', 'TULKAVR', 'TULKDK1', 'TULKDK2', 'TULKDK3', 'TULKDK4', 'TULKDK5', 'TULKDK6', 'TULKM2', 'TULKM3', 'TULKM4', 'TULKM5', 'TULKM6', 'TULKPS1', 'TULKPS2', 'TULKPS3', 'TULKPS4', 'TULKPS5', 'TULKPS6', 'TURETOT', 'TUSPABS', 'TUSPUSFT', 'TUSPWK', 'TXABSRSN', 'TXERN', 'TXERNH1O', 'TXERNH2', 'TXERNHRO', 'TXERNHRY', 'TXERNPER', 'TXERNRT', 'TXERNUOT', 'TXERNWKP', 'TXHRFTPT', 'TXHRUSL1', 'TXHRUSL2', 'TXHRUSLT', 'TXIO1COW', 'TXIO1ICD', 'TXIO1OCD', 'TXLAYAVL', 'TXLAYLK', 'TXLFS', 'TXLKAVL', 'TXLKM1', 'TXMJOT', 'TXRET1', 'TXSCHENR', 'TXSCHFT', 'TXSCHLVL', 'TXSPEMPNOT', 'TXSPUHRS', 'TXTCC', 'TXTCCTOT', 'TXTCOC', 'TXTHH', 'TXTNOHH', 'TXTO', 'TXTOHH', 'TXTONHH', 'TEAGE', 'TESEX', '10101', '10102', '10199', '10201', '10299', '10301', '10399', '10401', '20101', '20102', '20103', '20104', '20199', '20201', '20202', '20203', '20299', '20301', '20302', '20303', '20399', '20401', '20402', '20499', '20501', '20502', '20601', '20602', '20699', '20701', '20799', '20801', '20899', '20901', '20902', '20903', '20904', '20905', '20999', '29999', '30101', '30102', '30103', '30104', '30105', '30106', '30108', '30109', '30110', '30111', '30112', '30199', '30201', '30202', '30203', '30204', '30299', '30301', '30302', '30303', '30399', '30401', '30402', '30403', '30404', '30405', '30499', '30501', '30502', '30503', '30504', '30599', '40101', '40102', '40103', '40104', '40105', '40106', '40108', '40109', '40110', '40111', '40112', '40199', '40201', '40202', '40301', '40302', '40303', '40399', '40401', '40402', '40403', '40404', '40405', '40499', '40501', '40502', '40503', '40504', '40505', '40506', '40507', '40508', '40599', '49999', '50101', '50102', '50103', '50104', '50199', '50201', '50202', '50203', '50299', '50301', '50302', '50303', '50304', '50399', '50401', '50403', '50404', '50499', '59999', '60101', '60102', '60103', '60104', '60199', '60201', '60202', '60203', '60204', '60299', '60301', '60302', '60399', '60401', '60402', '60499', '69999', '70101', '70102', '70103', '70104', '70105', '70201', '80101', '80102', '80201', '80202', '80203', '80301', '80401', '80402', '80403', '80499', '80501', '80502', '80601', '80701', '80702', '89999', '90101', '90102', '90103', '90104', '90199', '90201', '90202', '90299', '90301', '90302', '90399', '90401', '90501', '90502', '90599', '99999', '100101', '100102', '100103', '100199', '100201', '100299', '100304', '100305', '100401', '110101', '110201', '120101', '120201', '120202', '120299', '120301', '120302', '120303', '120304', '120305', '120306', '120307', '120308', '120309', '120310', '120311', '120312', '120313', '120399', '120401', '120402', '120403', '120404', '120405', '120499', '120501', '120502', '120503', '120504', '130101', '130102', '130103', '130104', '130105', '130106', '130107', '130108', '130109', '130110', '130112', '130113', '130114', '130115', '130116', '130117', '130118', '130119', '130120', '130122', '130124', '130125', '130126', '130127', '130128', '130129', '130130', '130131', '130132', '130133', '130134', '130135', '130136', '130199', '130202', '130203', '130207', '130210', '130212', '130213', '130216', '130218', '130219', '130220', '130222', '130223', '130224', '130225', '130226', '130227', '130229', '130232', '130299', '130301', '130302', '139999', '140101', '140102', '140103', '140105', '149999', '150101', '150102', '150103', '150104', '150105', '150106', '150199', '150201', '150202', '150203', '150204', '150299', '150301', '150302', '150399', '150401', '150402', '150501', '150601', '150602', '150699', '150701', '150801', '159999', '160101', '160102', '160103', '160104', '160105', '160106', '160107', '160108', '160199', '160201', '180101', '180201', '180202', '180203', '180204', '180205', '180206', '180207', '180208', '180209', '180299', '180301', '180302', '180303', '180304', '180305', '180401', '180402', '180403', '180404', '180405', '180499', '180501', '180502', '180503', '180504', '180599', '180601', '180602', '180603', '180604', '180699', '180701', '180702', '180703', '180704', '180801', '180802', '180803', '180804', '180805', '180806', '180807', '180899', '180901', '180902', '180903', '180904', '180905', '180999', '181001', '181002', '181101', '181201', '181202', '181203', '181204', '181205', '181299', '181301', '181302', '181399', '181401', '181501', '181599', '181601', '181801', '189999', '500101', '500103', '500105', '500106', '500107']

if __name__ == "__main__":
	main()