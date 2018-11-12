import numpy as np;
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesClassifier,RandomTreesEmbedding
from sklearn.manifold import TSNE 
from sklearn.externals import joblib;
from sklearn.datasets import make_blobs;
from collections import Counter;
from multiprocessing import Pool;

def treeRefit(vector, labels,depth=None,nest = 500):
	new_maxleaf=None;
	new_sample_size = 1
	new_split_size = 2
	new_n_est = nest
	
	new_maxdepth = depth

	clf = ExtraTreesClassifier(n_jobs=1,max_leaf_nodes=new_maxleaf,n_estimators=new_n_est,criterion='entropy',min_samples_split=new_split_size,min_samples_leaf=new_sample_size,max_depth=new_maxdepth);
	
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

	clf = RandomTreesEmbedding(max_leaf_nodes=maxleaf,n_estimators=n_est,min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,min_impurity_decrease=maximpure)
	clf = clf.fit(vector);
	labels = clf.apply(vector);
	return clf, labels;

def tsneLabelFit(prox,perplex=10):
	tsne = TSNE(n_components = 2,perplexity=perplex,early_exaggeration=10.0,verbose=2,metric='precomputed')
	coords = tsne.fit_transform(1-prox)
	coords /= np.max(np.abs(coords));
	return coords;

def dbscanLabelFit(coords, eps = 0.05, samples = 10):
	dbscan = DBSCAN(eps = eps, min_samples= samples);
	dblabels = dbscan.fit_predict(coords);
	return dblabels;

def outerequal(x):
	tcount = len(x[0]);
	prox = np.zeros( (tcount,tcount) );
	for i in range(len(x)):
		prox += np.equal.outer(x[i],x[i]);

	return prox;


#per http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#workings,
#there is a means to use a two class method to estimate classes
#instead we can just use the proximity matrix method with random trees
#without the two class split, especially where there are a high number of components
#TSNE appears to work better than MDS in this case, but YMMV. 
def proxMat(labels,procs=8):
	_procs = procs
	if _procs < 1: _procs = 1;
	pool = Pool(_procs);
	
	lines = labels.T;
	split = np.array_split(lines,_procs);

	tcount = len(labels);
	prox = np.zeros( (tcount,tcount) );
	result = pool.map(outerequal, split);
	for i in range(len(result)):
		prox += result[i];
	pool.close();

	n_est = len(labels[0]);
	prox = prox / float(n_est);

	return prox

def labelReduce(labels):
	labelset = set(labels);
	labelmap = { item:index for index,item in enumerate(labelset) }
	newlabels = np.array([labelmap[b] for b in labels]);
	return newlabels;


def labelCutoff(vector, labels, cutoff):
	counts = Counter(labels);
	relabelfunc = np.vectorize(lambda x: x if counts[x] >= cutoff else -1);
	relabel = relabelfunc(labels);
	relabelvector = vector[relabel > -1];
	relabel = relabel[relabel > -1];
	return relabelvector, relabel;

def stripUnlabelled(vector,labels):
	stripframe = vector[labels > -1]
	striplabel = labels[labels > -1]
	return stripframe, striplabel;

#FIXME need to split to that tsne happens and allows eps/samples to be determined 
def determineCoords(vector, perplex=30,procs=4,nest=2000):

	# print("Initial random forest fitting...")
	initclf, initlabels = randoTrees(vector,depth=5,nest=nest);
	prox = proxMat(initlabels,procs=procs)
	coords = tsneLabelFit(prox,perplex=perplex)

	return coords;


def genLabels(vector,xycoords, n_cutoff = 25, eps=0.03, samples=20,nest = 2000):

	initlabels = dbscanLabelFit(xycoords, eps = eps, samples = samples)

	stripvector, striplabels = stripUnlabelled(vector, initlabels)
	refitclf,refitlabels = treeRefit(stripvector, striplabels, depth=8, nest=nest);
	refitlabels = refitclf.predict(vector);

	reducedvector, reducedlabels = labelCutoff(vector, refitlabels, n_cutoff);
	reduceclf,reducedlabels=treeRefit(reducedvector,reducedlabels,depth=None, nest=nest);
	reducedlabels = reduceclf.predict(vector);

	return reducedlabels, reduceclf;

def saveCLF(clf,path,name):
	joblib.dump(clf, path+"clfsave-"+name+".pkl", compress=('gzip',9));



def runtest():
	print("Gen vectors with 10 labels...")
	vector,truelabels = make_blobs(n_samples=1000,n_features=50,centers = 10,random_state=0)
	print("True label counts: ",Counter(truelabels))

	print("Getting coords...")
	coords = determineCoords(vector,nest=250)

	print("Reducing labels...")
	labels,clf = genLabels(vector,coords,n_cutoff=5,nest=250)
	labelcount = Counter(labels)
	print("Total final label count: ", len(labelcount))
	print("Reduced label counts: ",labelcount)

	print("Testing CLF save...")
	saveCLF(clf,"","test");


if(__name__ == "__main__"):
	runtest();