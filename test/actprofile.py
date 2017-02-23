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


# labels = dblabels;
# labels,labelsetcount = labelReduce(labels);
# labelscount = Counter(labels);


def treeRefit(vector, labels):
	new_maxleaf=None;
	new_sample_size = 1
	new_split_size = 2
	new_n_est = 3000
	new_maxdepth = 6

	clf = sklearn.ensemble.ExtraTreesClassifier(max_leaf_nodes=new_maxleaf,n_estimators=new_n_est,criterion='entropy',min_samples_split=new_split_size,min_samples_leaf=new_sample_size,max_depth=new_maxdepth);
	clf = clf.fit(supervector,labels);
	newlabels = clf.predict(supervector);
	return clf, newlabels;

def randoTrees(vector):
	split_size = 2
	sample_size = 1
	maximpure = 0.00001
	maxleaf = None
	n_est = 3000
	maxdepth = 6

	clf = sklearn.ensemble.RandomTreesEmbedding(max_leaf_nodes=maxleaf,n_estimators=n_est,min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,min_impurity_split=maximpure)
	clf = clf.fit(supervector);
	labels = clf.apply(supervector);
	return clf, labels;

def tsneLabelFit(prox):
	ncomp = 2;
	tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=2,early_exaggeration=100,verbose=2,metric='precomputed')
	coords = tsne.fit_transform(1-prox)
	dbscan = sklearn.cluster.DBSCAN(eps = 0.5, min_samples= 1);
	dblabels = dbscan.fit_predict(coords);
	return coords,labels;


def compPlot(coords, labels, prefix):
	#cmapp = np.linspace(0.0,1.0,10);
	ncomp = len(coords);
	cmapp = np.linspace(0.0,1.0,len(labels));
	colors = [ cm.jet(x) for x in cmapp ]
	np.random.shuffle(colors);
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
    labelset = set(labecode --wls);
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


#weekdayset = [2,3,4,5,6]
#weekdayset = [1,1]
weekdayset = [7,7]
dayselectset = weekdayset

jointable['mapped'] = jointable['TRCODE'].apply(lambda x: tri[x]);
#jointable = jointable.iloc[]
jointable = jointable[jointable['TUDIARYDAY'].apply(lambda x: x in dayselectset) ];
infotable = infotable[infotable['TUDIARYDAY'].apply(lambda x: x in dayselectset) ];

print('Processing...')
cases = jointable.groupby(['TUCASEID']) 
casecount = len(cases);

print(casecount)

#,'TRTALONE','TERET1','TRERNHLY','TRERNWA','TEIO1OCD',
#badcols = ['TEERNPER','TRHERNAL','TRTALONE_WK','TEIO1OICD','TEIO1OCD','TEIO1ICD','TRDTOCC1','TUYEAR','TUMONTH','TULINENO', 'TUDIARYDAY', 'TUDIARYDATE','TUFINLWGT','TUCASEID','TXABSRSN', 'TXERN', 'TXERNH1O', 'TXERNH2', 'TXERNHRO', 'TXERNHRY', 'TXERNPER', 'TXERNRT', 'TXERNUOT', 'TXERNWKP', 'TXHRFTPT', 'TXHRUSL1', 'TXHRUSL2', 'TXHRUSLT', 'TXIO1COW', 'TXIO1ICD', 'TXIO1OCD', 'TXLAYAVL', 'TXLAYLK', 'TXLFS', 'TXLKAVL', 'TXLKM1', 'TXMJOT', 'TXRET1', 'TXSCHENR', 'TXSCHFT', 'TXSCHLVL', 'TXSPEMPNOT', 'TXSPUHRS', 'TXTCC', 'TXTCCTOT', 'TXTCOC', 'TXTHH', 'TXTNOHH', 'TXTO', 'TXTOHH', 'TXTONHH']
#badcols = ['TUYEAR','TUMONTH','TULINENO', 'TUDIARYDAY', 'TUDIARYDATE','TUFINLWGT','TUCASEID']
#infocolumns = [col for col in infotable.columns if col not in badcols];
goodcols = ['TEAGE','TELFS','TESCHENR','TESCHFT','TESCHLVL','TESEX','TESPEMPNOT','TESPUHRS','TRCHILDNUM','TRDPFTPT','TRHHCHILD','TRSPPRES','TRTALONE','TRTCC','TRTCCC','TRTCCC_WK','TRTCOC','TRTCHILD','TRTFAMILY','TRTHH','TRTHHFAMILY','TRTNOCHILD','TRTNOHH','TRTO','TRTOHH','TRTONHH','TRTONHHCHILD','TRTSPONLY','TRTSPOUSE','TRTUNMPART','TUBUS','TUCC2','TUCC4','TUDIS','TUDIS1','TUDIS2','TUELDER','TUELNUM','TUELFREQ','TUFWK','TUSPABS','TUSPUSFT','TUSPWK']
infocolumns = [col for col in infotable.columns if col in goodcols];
casecol = pd.DataFrame(infotable['TUCASEID']);


#print(casecol);

print(infocolumns) 

#exit();

vectors = np.zeros((casecount,actcount))
vectors2 = np.zeros_like(infotable[infocolumns].values);
labels = np.zeros((casecount,),dtype=np.int64);
counts = np.zeros(actcount, dtype=np.uint32);
#secondorder = np.zeros((actcount+10,actcount), dtype=np.uint32);

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
#instead we can just use the proximity matrix method with random trees
#without the splits, especially where there are a high number of components
#TSNE works better than MDS in this case, but YMMV. 
clf, labels = randoTrees(vector);
prox = proxMat(labels);
coords, dblabels = tsneLabelFit(prox);

dbcount = Counter(dblabels)
print("DBSCAN labels",dbcount);
casecol['labels']=dblabels;

compPlot(coords.T,dblabels,imgpath+"/dbscan");

print("Random forest re-fitting...")
refitclf,newlabels = treeRefit(supervector, dblabels);

newlabelscount = Counter(newlabels);
print("New label count: ",newlabelscount);
casecol['newlabels'] = newlabels;


#second DBscan print
compPlot(coords.T, newlabels, imgpath+"/dbscan-new");
# casecol['newlabels'] = labels;
# newlabels = labels
# newlabelscount = Counter(newlabels);
	
newjoin = pd.merge(jointable,casecol,on='TUCASEID')


casecol.to_csv(imgpath+"/labels.csv")

print("secondary processing...")


clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2,min_samples_leaf=5,max_depth=8)
clf = clf.fit(supervector,newlabels);
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
	
	#print(njg[1])
	subsetnjg = njg[1][njg[1]['TUACTIVITY_N'] == 1]
	mdem = subsetnjg.groupby(["TESEX","TEAGE"]).TESEX.count()[1]
	fdem = subsetnjg.groupby(["TESEX","TEAGE"]).TESEX.count()[2]
	#print(mdem,fdem);
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


