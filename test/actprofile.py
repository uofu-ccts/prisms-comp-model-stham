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

def demoActPlot(frame,labelcolumn, prefix):
	#prelim stuff
	mapping = np.sort(list(set(jointable['TRCODE'])))
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
		
		if(labelscount[jg[0]] < 25):
			continue;
		
		daycount = np.zeros(7)
		data = np.zeros([actcount,288*7])
		sum = np.zeros(288*7)
		dayset = {1}
		
		#print(jg[1])
		subsetjg = jg[1][jg[1]['TUACTIVITY_N'] == 1]
		mdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[1]
		fdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[2]
		#print(mdem,fdem);
		if(type(mdem) == pd.core.series.Series): mdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='b')
		else: print("bad mdem:",mdem,type(mdem))
		if(type(fdem) == pd.core.series.Series): fdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='r')
		else: print("bad fdem:",fdem,type(fdem))

		plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		F = plt.gcf();
		F.set_size_inches(4,3)
		F.set_dpi(300);
		F.savefig(prefix + "demo-" +str(jg[0]) + ".png");
		plt.clf();
		
		
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
		
		print("plotting label "+ str(jg[0]) + ", c:"+str(labelscount[jg[0]]))
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

		plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
		plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		F = plt.gcf();
		F.set_size_inches(12,8)
		F.set_dpi(300);
		F.savefig(prefix +"label-" +str(jg[0]) + ".png");
		plt.clf();




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

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
tusepath = datapath + "timeuse/"

print("reading...")
acttable = pd.read_csv(tusepath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(tusepath + "atusresp_2015/atusresp_2015.dat")
rosttable = pd.read_csv(tusepath + "atusrost_2015/atusrost_2015.dat")

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

labelscount = Counter(newlabels);
print("New label count: ",labelscount);
casecol['newlabels'] = newlabels;


#second DBscan print
compPlot(coords.T, newlabels, imgpath+"/dbscan-new");
# casecol['newlabels'] = labels;
# newlabels = labels
# labelscount = Counter(newlabels);
	
newjoin = pd.merge(jointable,casecol,on='TUCASEID')

casecol.to_csv(imgpath+"/labels.csv")

print("secondary processing...")

clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2,min_samples_leaf=5,max_depth=8)
clf = clf.fit(supervector,newlabels);
writeTree(imgpath + "/dtreefinal.png",clf,[str(b) for b in mapping]+infocolumns)

def demoActPlot(newjoin, "newlabels", imgpath +"/")


