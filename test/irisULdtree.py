import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;
import sklearn.decomposition
import sklearn.cluster
import sklearn.ensemble
import sklearn.tree
import sklearn.datasets
import pydotplus;
from collections import Counter;
import re;
import os;


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

#Data Loading
iris = sklearn.datasets.load_iris()

samplecount = len(iris.target);
initlabels = np.arange(samplecount)
# initlabels = np.zeros(samplecount);
# secondlabels = np.ones(samplecount);
mapping = iris.feature_names

#Initial label guesses
print("Decision tree fitting...")


# secondclass = iris.data;
# secondclass = secondclass.T
# for i in range(len(secondclass)):
# 	np.random.shuffle(secondclass[i]);
# secondclass = secondclass.T

# pseudodata = np.concatenate((iris.data,secondclass));
# pseudolabels = np.concatenate((initlabels, secondlabels));

split_size = 2
sample_size = 1
maximpure = 0.00001
maxleaf = None
n_est = 500
maxdepth = 10

# clf = sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=maxleaf,n_estimators=n_est,criterion='entropy',min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,oob_score=True,min_impurity_split=maximpure)



# clf = sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=split_size,min_samples_leaf=sample_size,min_impurity_split=maximpure,max_leaf_nodes=maxleaf)

# clf = clf.fit(iris.data,initlabels);

clf = sklearn.ensemble.RandomTreesEmbedding(max_leaf_nodes=maxleaf,n_estimators=n_est,min_samples_split=split_size,min_samples_leaf=sample_size,max_depth=maxdepth,min_impurity_split=maximpure)


#clf = clf.fit(pseudodata,pseudolabels);
# print("Init OOB score: ",clf.oob_score_)

labels = clf.fit_transform(iris.data);

#print(newvec)
#print(type(newvec))

#writeTree("irisdtree-init.png",clf,mapping)

labels = clf.apply(iris.data);
labels = labels/np.amax(labels);

prox = proxMat(labels);
# plt.matshow((prox));
# plt.colorbar();
# plt.show();

tncomp = 10
ncomp = 5;
# # #mds = sklearn.manifold.MDS(n_components = ncomp, dissimilarity="precomputed",metric=False,verbose=2)
# # #coords = mds.fit_transform((1.0-prox)).T

svd = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
coords = svd.fit_transform(labels).T;

# svd = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
# coords2 = svd.fit_transform(labels);

# tsne = sklearn.manifold.TSNE(n_components = ncomp,perplexity=2,early_exaggeration=100,verbose=2,metric='precomputed')
# coords = tsne.fit_transform(1-prox).T




dbscan = sklearn.cluster.DBSCAN(eps = 1.0, min_samples=10);
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
		plt.title("TSNE:"+","+str(i)+","+str(j))
		plt.show()
		# F = plt.gcf();
		# F.set_size_inches(8,6)
		# F.set_dpi(300);
		# F.savefig(imgpath +"/dbscan-" +str(step) + ".png");
		# plt.clf();


labels = dblabels;

labels,labelsetcount = labelReduce(labels);
labelscount = Counter(labels);
print("First label count: ",labelscount);

#Random forest revision
maxleaf=None;
new_sample_size = 1
new_split_size = 2
new_n_est = 1000
maxdepth = None


print("Random forest fitting...")
clf = sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=maxleaf,n_estimators=new_n_est,criterion='entropy',min_samples_split=new_split_size,min_samples_leaf=new_sample_size,oob_score=True,max_depth=maxdepth);
clf = clf.fit(iris.data,labels);
newlabels = clf.predict(iris.data);

print("Final OOB score: ",clf.oob_score_)

newlabelscount = Counter(newlabels);
print("New label count: ",newlabelscount);

compare = Counter([ (b,t) for b,t in zip(newlabels,iris.target) ])

print("Comparison:",compare);