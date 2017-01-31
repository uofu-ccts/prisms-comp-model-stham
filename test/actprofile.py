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
import pydotplus;

from collections import Counter;
import re;
import os;

np.set_printoptions(threshold=np.inf)

def writeTree(name, clf, mapping):
    dotdata = sklearn.tree.export_graphviz(clf, feature_names=[str(b) for b in mapping],out_file=None, rotate=True,class_names=True,node_ids=True);
    dotdata = re.sub(r"value = \[([^]]+)\]", "", dotdata);
    #print(dotdata)
    g = pydotplus.graph_from_dot_data(dotdata);
    g.write_png(name);


def proxMat(labels):
    print("Calculating proximity matrix...")
    tcount = len(labels);
    n_est = len(labels[0]);

    prox = np.zeros( (tcount,tcount) );

    lines = labels.T;

    for i in range(len(lines)):
        #if(i%100==0): print(i,end="..",flush=True);
        prox += np.equal.outer(lines[i],lines[i]);
    #print("")
    prox = prox / float(n_est);

    print("Proximity percentiles...")
    print([0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99])
    print(np.percentile(prox,[0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99]))
    return prox

def labelReduce(labels):
    labelset = set(labels);
    labelmap = { item:index for index,item in enumerate(labelset) }
    newlabels = [labelmap[b] for b in labels];
    return newlabels,len(labelset);

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
rosttable = pd.read_csv(datapath + "atusrost_2015/atusrost_2015.dat")

print("fixing roster...");

rosttable = rosttable[rosttable['TERRP'].apply(lambda x: x in [18,19])]
# rosttable['TEAGE'] = rosttable['TEAGE'].apply(lambda x: x // 5);
rosttable = rosttable.drop(['TXAGE','TXRRP','TXSEX','TULINENO','TERRP'],axis=1)


print("joining...")
infotable = pd.merge(infotable,rosttable,on='TUCASEID')
jointable = pd.merge(acttable,infotable,on='TUCASEID')

#jointable = pd.merge(jointable, rosttable, on='TUCASEID');

#force some cleanliness:
infotable['TUCC2'] = 0
infotable['TUCC4'] = 0

mapping = np.sort(list(set(jointable['TRCODE'])))
#print([(i,k) for i,k in enumerate(mapping)]);
actcount = len(mapping)

tri = { tr:i for i,tr in enumerate(mapping) }
itr = { i:tr for i,tr in enumerate(mapping) }


weekdayset = [2,3,4,5,6]
dayselectset = [2,3,4,5,6]

jointable['mapped'] = jointable['TRCODE'].apply(lambda x: tri[x]);
#jointable = jointable.iloc[]
jointable = jointable[jointable['TUDIARYDAY'].apply(lambda x: x in dayselectset) ];
infotable = infotable[infotable['TUDIARYDAY'].apply(lambda x: x in dayselectset) ];

print('Processing...')
cases = jointable.groupby(['TUCASEID']) 
casecount = len(cases);

#,'TRTALONE','TERET1','TRERNHLY','TRERNWA','TEIO1OCD',
badcols = ['TRHERNAL','TRTALONE_WK','TEIO1OICD','TEIO1OCD','TEIO1ICD','TRDTOCC1','TUYEAR','TUMONTH','TULINENO', 'TUDIARYDAY', 'TUDIARYDATE','TUFINLWGT','TUCASEID','TXABSRSN', 'TXERN', 'TXERNH1O', 'TXERNH2', 'TXERNHRO', 'TXERNHRY', 'TXERNPER', 'TXERNRT', 'TXERNUOT', 'TXERNWKP', 'TXHRFTPT', 'TXHRUSL1', 'TXHRUSL2', 'TXHRUSLT', 'TXIO1COW', 'TXIO1ICD', 'TXIO1OCD', 'TXLAYAVL', 'TXLAYLK', 'TXLFS', 'TXLKAVL', 'TXLKM1', 'TXMJOT', 'TXRET1', 'TXSCHENR', 'TXSCHFT', 'TXSCHLVL', 'TXSPEMPNOT', 'TXSPUHRS', 'TXTCC', 'TXTCCTOT', 'TXTCOC', 'TXTHH', 'TXTNOHH', 'TXTO', 'TXTOHH', 'TXTONHH']
#badcols = ['TUYEAR','TUMONTH','TULINENO', 'TUDIARYDAY', 'TUDIARYDATE','TUFINLWGT','TUCASEID']
infocolumns = [col for col in infotable.columns if col not in badcols];
casecol = pd.DataFrame(infotable['TUCASEID']);


#print(casecol);

# goodcols = [ 'TEHRUSLT','TRTFAMILY','TRNUMHOU','TRTHHFAMILY','TRTCHILD','TRTOHHCHILD','TELFS','TRTCCTOT','TRTALONE' ]
# 
# infocolumns = [col for col in infocolumns if col in goodcols];

#print(infocolumns) 

#exit();

vectors = np.zeros((casecount,actcount))
vectors2 = np.zeros_like(infotable[infocolumns].values);
labels = np.zeros((casecount,),dtype=np.int64);
counts = np.zeros(actcount, dtype=np.uint32);
secondorder = np.zeros((actcount+10,actcount), dtype=np.uint32);

randlabel = 3;


for ind,i in enumerate(cases):
	t = list(set(i[1]['mapped']))
	vectors[ind][t] += 1;
	counts[t] += 1;
	#print(i)
	#print(i[1]['TUCASEID'][0]);
	#print(infotable[infotable['TUCASEID']==i[1]['TUCASEID'][0]]);
	#print("next");
	vectors2[ind] = infotable[infotable['TUCASEID']==i[1]['TUCASEID'].iloc[0]][infocolumns].values;
	#print(i,vectors2[ind]);
	#labels[ind] = [i[0],i[1]['TUDIARYDATE'].iloc[0]]
	#labels[ind] = np.random.randint(0,randlabel);
	labels[ind] = ind
	#exit();


scount = np.sort(counts)

actcutoff = 300;

thresh = scount[-actcutoff];

scount = np.arange(0,actcount)[counts >= thresh];

#print([(b,itr[b]) for b in scount])

# lookup = {};
# lookupind = 0;
# 
# for ind,i in enumerate(vectors):
# 	key = tuple(i[scount])
# 	if key not in lookup:
# 		lookup[key] = lookupind;
# 		lookupind += 1;
# 	labels[ind] = lookup[key];
# 
# print("Label count:", lookupind);

print("Casecount:", casecount);

imgpath = time.strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(imgpath)

print("Initial random forest fitting...")
supervector = np.concatenate((vectors,vectors2),axis=1)

#per http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#workings,
#there is a means to use a two class method to estimate effectiveness

# secondvector = supervector.T
# for i in range(len(secondvector)):
# 	np.random.shuffle(secondvector[i]);
# secondvector = secondvector.T;
# secondlabels = np.ones((casecount,),dtype=np.int64);

# unsupervector = np.concatenate((supervector,secondvector))
# unlabels = np.concatenate((labels, secondlabels));

split_size = 2
sample_size = 1
maximpure = 0.00001
maxleaf = None
n_est = 500
maxdepth = 10

# #first pass truncation for initial labelling
# #clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=80)
# # clf = sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=maxleaf,n_estimators=n_est,criterion='entropy',min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=None,oob_score=True,min_impurity_split=maximpure)
# #clf = clf.fit(unsupervector,unlabels);
# clf = clf.fit(supervector,labels);

clf = sklearn.ensemble.RandomTreesEmbedding(max_leaf_nodes=maxleaf,n_estimators=n_est,min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,min_impurity_split=maximpure)


#clf = clf.fit(pseudodata,pseudolabels);
# print("Init OOB score: ",clf.oob_score_)
labels = clf.fit_transform(supervector);
labels = clf.apply(supervector);
prox = proxMat(labels);

# labels = labels/np.amax(labels);


ncomp = 2;

# svd = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
# coords = svd.fit_transform(labels).T;

tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=2,early_exaggeration=100,verbose=2,metric='precomputed')
coords = tsne.fit_transform(1-prox).T


dbscan = sklearn.cluster.DBSCAN(eps = 0.5, min_samples=3);
dblabels = dbscan.fit_predict(coords.T);
dbcount = Counter(dblabels)
print("DBSCAN labels",dbcount);

cmapp = np.linspace(0.0,1.0,len(dblabels));
#cmapp = np.linspace(0.0,1.0,densitysteps);
colors = [ cm.jet(x) for x in cmapp ]
np.random.shuffle(colors);
outc = [ colors[b] if b > -1 else (0,0,0,1) for b in dblabels ];
for i in range(ncomp):
	for j in range(i+1,ncomp):
		plt.scatter(coords[i],coords[j],color=outc,s=8,edgecolor='');
		plt.title("TSVD:"+","+str(i)+","+str(j))
		# plt.show()
		F = plt.gcf();
		F.set_size_inches(8,6)
		F.set_dpi(300);
		F.savefig(imgpath +"/dbscan-" +str(i)+"-"+str(j)+ ".png");
		plt.clf();


labels = dblabels;

labels,labelsetcount = labelReduce(labels);
labelscount = Counter(labels);
#print("First label count: ",labelscount);

# exit()

#print(newvec)
#print(type(newvec))

#writeTree("irisdtree-init.png",clf,mapping)




# print("Init OOB score: ",clf.oob_score_)
# labels = clf.apply(supervector);


# for ind,i in enumerate(clf.estimators_):
# 	dotdata = sklearn.tree.export_graphviz(i, feature_names=[str(b) for b in mapping]+infocolumns,out_file=None, rotate=True,class_names=True,node_ids=True);
# 	dotdata = re.sub(r"value = \[([^]]+)\]", "", dotdata);
# 	#print(dotdata)
# 	g = pydotplus.graph_from_dot_data(dotdata);
# 	g.write_png(imgpath+"/dtreeinit-"+str(ind)+".png");
# 	break;


# tcount = len(labels);

# print("Prox mat time...")
# prox = np.zeros( (tcount,tcount) );



# lines = labels.T;

# for i in range(len(lines)):
# 	if(i%100==0): print(i,end="..",flush=True);
# 	prox += np.equal.outer(lines[i],lines[i]);
# print("")
# prox = prox / float(n_est);

# print([0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99])
# print(np.percentile(prox,[0.1,1.,5.,10.,25.,50.,75.,90.,95.,99.,99.9,99.99]))

# cutoff = np.percentile(prox,99.9)

# prox[prox < cutoff] = 0.0

# plt.matshow((prox));
# plt.colorbar();
# F = plt.gcf();
# F.set_size_inches(10,8)
# F.set_dpi(300);
# F.savefig(imgpath +"/prox.png");
# plt.clf()
# plt.show();

# exit()



# print(dbcount)

# print("MDS compute...")

# ncomp = 2

# mds = sklearn.manifold.MDS(n_components = ncomp, dissimilarity="precomputed",metric=True)

# coords = mds.fit_transform((1.0-prox)).T

# for i in range(ncomp):
# 	for j in range(i+1,ncomp):
# 		plt.scatter(coords[i],coords[j],color='k',s=8,edgecolor='');
# 		plt.title("MDS:"+str(i)+","+str(j))
# 		plt.show()

# print("TSNE compute...")


# step = 0;
# #for step in range(1):
# print("Step ", step);
# ncomp = 2;
# tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=1,early_exaggeration=100,verbose=2, metric="precomputed")
# #tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=1,early_exaggeration=100,verbose=2,init='pca')


# coords = tsne.fit_transform((1-prox)).T
# #coords = tsne.fit_transform(labels).T


# dbscan = sklearn.cluster.DBSCAN(eps = 0.30, min_samples=5);

# dblabels = dbscan.fit_predict(coords.T);

# dbcount = Counter(dblabels)
# print(dbcount)

# cmapp = np.linspace(0.0,1.0,len(dblabels));
# #cmapp = np.linspace(0.0,1.0,densitysteps);
# colors = [ cm.jet(x) for x in cmapp ]
# np.random.shuffle(colors);
# outc = [ colors[b] if b > -1 else (0,0,0,1) for b in dblabels ];

# for i in range(ncomp):
# 	for j in range(i+1,ncomp):
# 		plt.scatter(coords[i],coords[j],color=outc,s=8,edgecolor='');
# 		plt.title("TSNE:"+str(step)+","+str(i)+","+str(j))
# 		#plt.show()
# 		F = plt.gcf();
# 		F.set_size_inches(8,6)
# 		F.set_dpi(300);
# 		F.savefig(imgpath +"/dbscan-" +str(step) + ".png");
# 		plt.clf();

# # exit()
# #labels = [tuple(b) for b in labels]


# labels = dblabels;
# labelset = set(labels);
# #print(labelset)
# labelmap = { item:index for index,item in enumerate(labelset) }
# labelmap[-1]=-1
# #print(labelmap);
# labels = [labelmap[b] for b in labels];
# weights = [0.0001 if b == -1 else 1.0 for b in labels];

# unilabels = max(labels);
# casecol['labels'] = labels
# labelscount = Counter(labels);



# # dotdata = sklearn.tree.export_graphviz(clf, feature_names=[str(b) for b in mapping]+infocolumns,out_file=None, rotate=True,class_names=True,node_ids=True);
# # dotdata = re.sub(r"value = \[([^]]+)\]", "", dotdata);
# # #print(dotdata)
# # g = pydotplus.graph_from_dot_data(dotdata);
# # g.write_png(imgpath+"/dtree-init.png");

# print("First label count: ",labelscount);



maxleaf=None;
new_sample_size = 50
new_split_size = 2
new_n_est = 1000

print("Random forest fitting...")
clf = sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=maxleaf,n_estimators=new_n_est,criterion='entropy',min_samples_split=new_split_size,min_samples_leaf=new_sample_size,oob_score=True);
clf = clf.fit(supervector,labels);
newlabels = clf.predict(supervector);

print("Final OOB score: ",clf.oob_score_)

uninewlabels = max(newlabels);
#newlabels = clf.apply(supervector);
# print(newlabels[0:10]) 
# newlabels = [tuple(b) for b in newlabels]
# print(newlabels[0:10]) 

newlabelscount = Counter(newlabels);
print("New label count: ",newlabelscount);
casecol['newlabels'] = newlabels;

# print("Plotting...")
# feat = clf.feature_importances_
# 
# s = np.argsort(feat);
# 
# fnames=[str(b) for b in mapping]+infocolumns
# 
# s = [(fnames[b],feat[b]) for b in s ]
# 
# print(s);
# plt.bar(np.arange(0,len(feat)),feat);
# plt.show();

#infomap = list(infotable.columns)




# for ind,i in enumerate(clf.estimators_):
# 	dotdata = sklearn.tree.export_graphviz(i, feature_names=[str(b) for b in mapping]+infocolumns,out_file=None, rotate=True,class_names=True,node_ids=True);
# 	dotdata = re.sub(r"value = \[([^]]+)\]", "", dotdata);
# 	#print(dotdata)
# 	g = pydotplus.graph_from_dot_data(dotdata);
# 	g.write_png(imgpath+"/dtree-"+str(ind)+".png");
# 	break;
	
newjoin = pd.merge(jointable,casecol,on='TUCASEID')


# diffmat = np.zeros((unilabels+1, uninewlabels+1));
# 
# for i in casecol.iterrows():
# 	x = i[1]['labels']
# 	y = i[1]['newlabels']
# 	diffmat[x][y] += 1;
# 
# plt.matshow(diffmat);
# plt.xlabel("old")
# plt.ylabel("new")
# plt.colorbar();
# plt.show();






print("secondary processing...")


clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2,min_samples_leaf=1,max_depth=8)
clf = clf.fit(supervector,labels);
writeTree(imgpath + "/dtreefinal.png",clf,[str(b) for b in mapping]+infocolumns)


start = 0.0
stop = 1.0
number_of_lines=19
cm_subsection = np.linspace(start, stop, number_of_lines)


maincolors = [ cm.prism(x) for x in cm_subsection ]

colors = []

for i in mapping:
	tier1 = i // 10000;
	tier2 = (i // 100) - (tier1 * 100)
	#print(tier2);
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
	 
#print(colors[0:10]);


legart = []
leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
for i in range(number_of_lines):
	legart += [mpatch.Rectangle((0,0),1,1,fc=maincolors[i])]

newjoingroup = newjoin.groupby('newlabels')
for njg in newjoingroup:
	
	if(newlabelscount[njg[0]] < 25):
		continue;
	
	daycount = np.zeros(7)
	data = np.zeros([actcount,288*7])
	sum = np.zeros(288*7)
	dayset = {1}
	
	print(njg[1])
	mdem = njg[1].groupby(["TESEX","TEAGE"]).TESEX.count()[1]
	fdem = njg[1].groupby(["TESEX","TEAGE"]).TESEX.count()[2]
	print(mdem,fdem);
	if(type(mdem) == pd.core.series.Series): mdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='b')
	else: print("bad mdem:",mdem,type(mdem))
	if(type(fdem) == pd.core.series.Series): fdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='r')
	else: print("bad fdem:",fdem,type(fdem))

	plt.title( "Label " +str(njg[0])+", count: "+str(newlabelscount[njg[0]]) )
	F = plt.gcf();
	F.set_size_inches(4,3)
	F.set_dpi(300);
	F.savefig(imgpath + "/demo-" +str(njg[0]) + ".png");
	plt.clf();
	 
	
	for ind, row in njg[1].iterrows():
		
		fullcode = tri[int(row['TRCODE'])]
		
# 		codetier1 = int(row['TUTIER1CODE']) - 1
# 		if(codetier1 > (18 - 1)):
# 	# 	if(codetier1 !=(17)):
# 			continue;
	# 	
	# 	codetier2 = int(row['TUTIER2CODE']) - 1;
	# 	if(codetier2 == 98): codetier2 = 17 - 1;
	# 	if(codetier2 > (18 - 1)):
	# 		continue;
	# 	 
		day =  int(row['TUDIARYDAY']) - 1;
		day =  (0 if int(row['TUDIARYDAY']) in weekdayset else 1);
		dayset.add(day);
		daycount[day] += 1;
		#t = time.strptime(row['TUSTARTTIM'],'%H:%M:%S')
		#start = np.floor((t.tm_hour*60+t.tm_min)/5)
		#t = time.strptime(row['TUSTOPTIME'],'%H:%M:%S')
		#stop = np.floor((t.tm_hour*60+t.tm_min)/5)
		
		
		stop = np.floor(row['TUCUMDUR24']/5.0);
		start = stop - np.floor(row['TUACTDUR24']/5.0)
	# 	if(start == stop): #implicitly, we don't expect someone to do the same thing for 24 hours straight unless they are messed up
	# 		stop = start + 1;
		startind = int(day * 288 + start)
		stopind = int(day * 288 + stop)
		
	# 	if(stop < start):
	# 		stopind += 288;
	# 		if(stopind > (day*288+288+48)):
	# 			stopind = day*288+288+48;
	
	
		data[fullcode,startind:stopind] += 1;
	# 	data[codetier2,startind:stopind] += 1;
	
		sum[startind:stopind] += 1;
		#else:
			
	# 		data[codetier1,startind:(day*288+288)] += 1;
	# 		data[codetier1,(day*288):stopind] += 1;
			
	#print("normalizing...")
	for i in range(len(data)):
		data[i] /= sum;
	
	
	#print("set of day vals: ",dayset);
	
	print("plotting label "+ str(njg[0]) + ", c:"+str(newlabelscount[njg[0]]))
	# x = np.arange(1.0,(8.0+(4.0/24.0)),(8.-1.0)/(288*7+48))
	x = np.arange(0.0,7.0,1.0/288.0)
	
	additive = np.zeros(288*7)
	
	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.set_ylim([0.0,1.0])
	

	datasum = np.cumsum(data, axis=0);
	ax.fill_between(x,0,datasum[0],facecolor=colors[0],linewidth=0.1)
	
	for i in range(len(colors)-1):
		ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],linewidth=0.1)
		#
	# 	additive += data[i];
	# 	plt.plot(x,datasum[i],linewidth=2.0,color=colors[i]);
	# 	plt.plot(x,data[i],color=c,linewidth=4.0);
	
	
	plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title( "Label " +str(njg[0])+", count: "+str(newlabelscount[njg[0]]) )
	#plt.show();
	F = plt.gcf();
	F.set_size_inches(12,8)
	F.set_dpi(300);
	F.savefig(imgpath + "/label-" +str(njg[0]) + ".png");
	plt.clf();


# compnum = 10
# 
# print('DECOMP time...')
# decomp = sklearn.decomposition.FastICA(n_components=compnum)#,kernel='sigmoid',n_jobs=2)
# #decomp2 = sklearn.decomposition.FastICA(n_components=compnum)#,kernel='sigmoid',n_jobs=2)
# 
# outvec = decomp.fit_transform(vectors);
# outvec = outvec.T;
# 
# #outvec2 = decomp2.fit_transform(vectors);
# #outvec2 = outvec2.T;
# 
# # print('plotting....')
# # # plt.plot(outvec[0],outvec[1],'o');
# # # plt.show();
# # 
# # plt.ylim(0.5,compnum+0.5);
# # for i in range(compnum):
# # 	min = np.min(outvec[i])
# # 	max = np.max(outvec[i]);
# # 	
# # 	plt.plot((outvec[i]-min)/(max-min),np.linspace(i+1,i+1,len(outvec[i])), '|');
# # 	
# # plt.show();
# # 
# # exit()
# densitysteps = 10;
# 
# nvec = int(compnum);
# print((nvec*(nvec+1))/2)
# overlabels = np.zeros( ( (nvec*(nvec+1))/2, casecount ) );
# c = 0;
# for i in range(nvec):
# 	#for j in range(i+1,len(outvec)):
# 	for j in range(i, nvec):	
# 		
# #DENSITY STEP MODE	
# 		munge = np.array([outvec[i],outvec[j]]).T
# 		mungecount = len(munge);
# 		#pick cell size
# 		ptp = np.ptp(munge,axis=0);
# 		area = ptp[0]*ptp[1];
# 		cellarea = area / mungecount;
# 		cellsize = np.sqrt(cellarea/np.pi)*3.0; #multiply by three to account for weird densities;
# 		print("Cellsize:",cellsize);
# 		labels=np.linspace(-1,-1,mungecount,dtype=np.int);
# 		basec = 0;
# 		density = np.ceil(np.logspace(np.ceil(np.log10(mungecount)),0.5,densitysteps))
# 		#density = np.ceil(np.linspace(mungecount,1,30));
# 		for ind,d in enumerate(density):
# 			clmode = sklearn.cluster.DBSCAN(n_jobs=1,min_samples=d,eps=cellsize);
# 			index = labels==-1;
# 			if(len(index) < 1): break;
# 			newlabels = clmode.fit_predict(munge[index])
# 			newlabels[newlabels > -1] += basec;
# 			#newlabels[newlabels > -1] = ind;
# 			basec += (len(np.unique(newlabels))-1)
# 			labels[index] = newlabels;
# 
# 		overlabels[c] = labels;
# 		c += 1
# 		unilabels = np.unique(labels);
# 		print(len(unilabels));
# 		cmapp = np.linspace(0.0,1.0,len(unilabels));
# 		#cmapp = np.linspace(0.0,1.0,densitysteps);
# 		colors = [ cm.plasma(x) for x in cmapp ]
# 		np.random.shuffle(colors);
# 		outc = [ colors[b] if b > -1 else (0,0,0,1) for b in labels ];
# 		plt.scatter(outvec[i],outvec[j],s=13,c=outc,marker='o',label=str(i)+','+str(j),alpha=0.7,edgecolors='none')
# 		plt.gca().set_aspect(1.0);
# 		plt.title(str(i)+','+str(j));
# 		plt.show();


# overlabels = overlabels.T
# counts = Counter([ (b[0],b[1]) for b in list(overlabels[)])
# print(counts);







#STRAIGHT UP DBSCAN
		
# 		munge = np.array([outvec[i],outvec[j]]).T
# 		mungecount = len(munge);
# 		#pick cell size
# 		ptp = np.ptp(munge,axis=0);
# 		area = ptp[0]*ptp[1];
# 		cellarea = area / mungecount;
# 		cellsize = np.sqrt(cellarea/np.pi)*3.0; #multiply by three to account for weird densities;
# 		print("Cellsize:",cellsize);
# 		
# 		clmode = sklearn.cluster.DBSCAN(n_jobs=1,min_samples=10,eps=cellsize);
# 		labels = clmode.fit_predict(munge);
# 		
# 		unilabels = np.unique(labels);
# 		print(len(unilabels));
# 		cmapp = np.linspace(0.0,1.0,len(unilabels));
# 		colors = [ cm.jet(x) for x in cmapp ]
# 		np.random.shuffle(colors);
# 		outc = [ colors[b] if b > -1 else (0,0,0,1) for b in labels ];
# 		plt.scatter(outvec[i],outvec[j],s=13,c=outc,marker='o',label=str(i)+','+str(j),alpha=0.6,edgecolors='none')
# # 		plt.gca().set_aspect(1.0);
# 		plt.title(str(i)+','+str(j));
# 		plt.show();



# shortlist = [3,25]
# for i in range(len(shortlist)):
# 	for j in range(i,len(shortlist)):
# 		if(i == j): continue;
# 		plt.plot(outvec[shortlist[i]],outvec[shortlist[j]],'o',label=str(i)+','+str(j),alpha=0.5)
# # 		if(i == j): continue;
# # 		#ax = plt.subplot(compnum,compnum,i*4+j+1);
# # 		plt.plot(outvec[i],outvec[j],'o');
# # 		plt.title(str(i)+','+str(j));
# # 		plt.show();
# plt.legend();
# plt.show();
# 
# print("Casecount:", casecount);
# 
# counts = np.zeros(actcount, dtype=np.uint32);
# secondorder = np.zeros((actcount+10,actcount), dtype=np.uint32);
# 
# for i in cases:
# 	t = list(set(i[1]['mapped']))	
# 	counts[t] += 1;
# 	for k in t:
# 		secondorder[k][t] += 1;
# 
# 
# counts = counts;
# secondorder = secondorder;
# secondorder[actcount:(actcount+10)] = counts;
# 
# # altmat = counts * counts.T;
# # np.fill_diagonal(altmat,counts)
# # diffmat = altmat-secondorder;
# 
# #print(counts);
# 
# # plt.bar(np.arange(0,actcount,1.0),counts/casecount);
# # plt.xticks(np.arange(0,actcount,1.0),(str(b) for b in mapping),rotation=45);
# # plt.show();
# 
# plt.matshow(secondorder/casecount);
# #plt.matshow(diffmat);
# plt.colorbar();
# plt.show();
