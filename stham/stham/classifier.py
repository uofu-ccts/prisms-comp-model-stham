




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
	tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=perplex,early_exaggeration=10.0,verbose=2,metric='precomputed')
	coords = tsne.fit_transform(1-prox)
	coords /= np.max(coords);
	dbscan = sklearn.cluster.DBSCAN(eps = eps, min_samples= samples);
	dblabels = dbscan.fit_predict(coords);
	return coords,dblabels;


def compPlot(coords, labels, prefix):
	neglabels = labels == -1
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
	prox = np.zeros( (tcount,tcount) );
	for i in range(len(x)):
		prox += np.equal.outer(x[i],x[i]);

	return prox;

def proxMat(labels,procs=8):
	print("Calculating proximity matrix...")
	tcount = len(labels);
	n_est = len(labels[0]);

	prox = np.zeros( (tcount,tcount) );

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
	
	for i in range(len(result)):
		prox += result[i];

	pool.close();

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



#FIXME
# def columnStrip(frame, columns):
# 	framecolumns = [col for col in frame.columns if col in columns];
# 	subframe = frame[framecolumns].copy();
# 	return subframe;

# def labelCutoff(frame, labels, cutoff):
# 	counts = Counter(labels);
# 	relabelfunc = np.vectorize(lambda x: x if counts[x] >= cutoff else -1);
# 	relabel = relabelfunc(labels);
# 	relabelframe = frame[relabel > -1];
# 	relabel = relabel[relabel > -1];
# 	return relabelframe, relabel;

# def stripUnlabelled(frame,labels,badlabel):
# 	stripframe = frame[labels > -1]
# 	striplabel = labels[labels > -1]
# 	return stripframe, striplabel;

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

	print("Initial random forest fitting...")
	initclf, initlabels = randoTrees(vector,depth=5,nest=2000);
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
