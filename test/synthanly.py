import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import sklearn
import sklearn.ensemble
from multiprocessing import Pool;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
from matplotlib.collections import PatchCollection;
import h5py
# import sqlite3

def plotseq(frame):
	fig, ax = plt.subplots()

	cneg = -1
	cpos = 1;

	# cl = np.linspace(0,1.0,386);
	# colors = [ cm.prism(x) for x in cl ]


	for i,gr in enumerate(frame.groupby("agentnum")):
		g,df = gr;
		# df = df.sort_values(["TUACTIVITY_N"])
		patches = [];


		# day =  int(df['TUDIARYDAY'][0]) - 1;
		# day =  (1 if df['TUDIARYDAY'].iloc[0] in weekdayset else 0);
		day = True;

		x = df['start'].values/60.0;
		y = np.zeros_like(x)

		if(day): 
			y[:] = cneg
			cneg -=1
		else:
			y[:] = cpos
			cpos += 1

		c = df["colors"].values;
		w = df['length'].values/60.0;
		h = np.ones_like(w)
		
		for xi,yi,wi,hi,ci in zip(x,y,w,h,c):
			patches.append(mpatch.Rectangle((xi,yi),wi,hi,facecolor=ci,edgecolor='black',linewidth=0.5))
		
		p = PatchCollection(patches,match_original=True)

		ax.add_collection(p)

		# cn+=1;
		# if(cn > 500): break;
		

	# fig.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	#print(cn);

	ax.set_xlim((0.,24.))
	ax.set_xticks((0,6,12,18,24))
	ax.set_xticklabels(("4:00", "10:00", "16:00", "22:00","4:00"))
	ax.set_yticks(());
	ax.set_ylim((cneg,cpos))
	# ax.set_ylim(-24.0,0.0)
	# ax.set_aspect(1.0)

	plt.show()

def outerequal(x):
	tcount = len(x[0]);
	prox = np.zeros( (tcount,tcount) );
	for i in range(len(x)):
		prox += np.equal.outer(x[i],x[i]);

	return prox;


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

def vectorize(fr):
	vec = np.zeros(1440);

	for ind,df in fr.iterrows():
		st = int(np.floor(df.start))
		en = int(np.floor(df.end))
		vec[st:en] = df.actind;

	return vec

def compareseq(v1,v2):
	return np.sum(v1 == v2)/1440.0

def getselfsim(v):
	similarities = np.zeros((len(v)*len(v) - len(v))//2)
	c = 0
	for i in range(len(v)):
		for j in range(i+1,len(v)):
			similarities[c] = compareseq(v[i],v[j])
			c+=1
	return similarities

def getcrosssim(v1,v2):
	similarities = np.zeros(len(v1)*len(v2))
	c = 0
	for i in range(len(v1)):
		for j in range(len(v2)):
			similarities[c] = compareseq(v1[i],v2[j])
			c += 1
	return similarities


def getrealseqs(actmapping):


	ati = { tr:i for i,tr in enumerate(actmapping) }
	actdatapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/" 

	acttable = pd.read_csv(actdatapath + "timeuse/atusact_2015/atusact_2015.dat")
	labels = pd.read_csv(actdatapath + "final-label-classifier/labels.csv")

	# acttable['start'] = (acttable['TUCUMDUR24']-acttable['TUACTDUR24'])//min_step
	# acttable['end'] = acttable['TUCUMDUR24']//min_step
	# acttable['length'] = acttable['end']-acttable['start']
	# acttable['length'] = acttable['length'].apply(lambda x:1 if x <= 0 else x)

	df = pd.DataFrame({"start":(acttable['TUCUMDUR24']-acttable['TUACTDUR24']),"end":acttable['TUCUMDUR24'],"actind":acttable["TRCODE"].apply(lambda x: ati[x]), "TUCASEID":acttable["TUCASEID"] })

	# df["daytype"] = labels["daytypelabelreduce"]

	df = pd.merge(df,labels[["TUCASEID","daytypelabelreduce"]],on='TUCASEID')
	df.rename(columns={"TUCASEID":"agentnum","daytypelabelreduce":"daytype"},inplace=True)



	return df;
	#need daytype, start, end, and actind


def procseqs(seqs):
	allvecs = {}
	c = 0
	for sind,sgf in seqs.groupby("daytype"):
		print(sind,end=' ')
		vecs = []

		# plotseq(sgf)


		for ind,gf in sgf.groupby("agentnum"):
			vecs += [[vectorize(gf)]]
		
		allvecs[sind] = np.concatenate(vecs,axis=0)
		# if(c > 3): break;
		c += 1
	return allvecs

def runit():

	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/" 
	simseqs = pd.read_csv(datapath + "simseqs.csv")

	prior = h5py.File(datapath + "actwindows.h5",'r');
	actmapping = prior["/actmapping"][:];
	prior.close();

	realseqs = getrealseqs(actmapping)
	# print(realseqs)
	# exit()

	start = 0.0
	stop = 1.0
	number_of_lines=19
	cm_subsection = np.linspace(start, stop, number_of_lines)
	# np.random.shuffle(cm_subsection)
	maincolors = [ cm.prism(x) for x in cm_subsection ]
	colors = []
	for i in actmapping:
		tier1 = i // 10000;
		if (tier1 == 50): tier1 = 19;
		tier1 = tier1 - 1;
		
		scol = list(maincolors[tier1])
	# 	scol[2] = scol[2] + (tier2 * 0.1)
	# 	if(scol[2] > 1.0): scol[2] = 1.0;
	# 	scol[3] = scol[3] + (tier2 * 0.1)
	# 	if(scol[3] > 1.0): scol[3] = 1.0;
		colors += [list(scol)];

	# ita = { i:tr for i,tr in enumerate(actmapping) }
	simseqs["colors"] = simseqs["actind"].apply(lambda x: colors[x])
	realseqs["colors"] = realseqs["actind"].apply(lambda x: colors[x])
	# simseqs0 = simseqs[simseqs["agentnum"] == 0]
	# simseqs1 = simseqs[simseqs["agentnum"] == 1]

	# v0 = vectorize(simseqs0)
	# v1 = vectorize(simseqs1)

	# print(compareseq(v0,v1))

	print("Vectorizing...")
	simvecs = procseqs(simseqs)
	realvecs = procseqs(realseqs)

		# print(np.shape(vecs))
		# c = 0
		# similarities = np.zeros((len(vecs)*len(vecs) - 500)//2)
		# for i in range(len(vecs)):
		# 	for j in range(i+1,len(vecs)):
		# 		similarities[c] = compareseq(vecs[i],vecs[j])
		# 		c+=1
	
	print("Finished vectorization...")
	avgs = np.zeros((6,len(simvecs.keys())))
	c = 0
	for p in simvecs.keys():
		print("Set: ",p)
		# clf, labels = randoTrees(np.concatenate([simvecs[p],realvecs[p]],axis=0),nest=2000,depth=10)
		simlen = len(simvecs[p])
		reallen = len(realvecs[p])
		# print(simlen,reallen)
		# prox = proxMat(labels,procs=2)

		# # plt.matshow(prox,vmin=0.0,vmax=1.0)
		# # plt.colorbar()
		# # plt.show()

		# similarities = [];
		# similarities += [prox[0:simlen,0:simlen][np.tril_indices(simlen,k=-1)]]
		# similarities += [prox[0:simlen,simlen:simlen+reallen]]
		# similarities += [prox[simlen:simlen+reallen,simlen:simlen+reallen][np.tril_indices(reallen,k=-1)]]
		similarities = []
		similarities += [getselfsim(simvecs[p])]
		similarities += [getcrosssim(simvecs[p],realvecs[p])]
		similarities += [getselfsim(realvecs[p])]
		# print(similarities)


		# for i in range(0,2):
		# 	for j in range(i,2):
				# similarities = prox[i*500:i*500+500,j*500:j*500+500][np.tril_indices(500,k=-1)]
				# similarities = prox
		for i in range(3):
			# print("Set: ", p);
			avgs[i,c] = np.mean(similarities[i])
			avgs[i+3,c] = np.std(similarities[i])
			print("Avg: ", avgs[i,c]," StdDev: ", avgs[i+3,c])
			# hist,edges = np.histogram(similarities[i],bins=50,range=(0.0,1.0),normed=True)
			# plt.plot(edges[:-1],hist,linewidth=0.5,label=str(p)+","+str(i))
		c+=1
	# plt.legend()
	# plt.show()



	avgframe = pd.DataFrame({"key":list(simvecs.keys()),"sim-sim":avgs[0],"sim-real":avgs[1],"real-real":avgs[2],"std-sim-sim":avgs[3],"std-sim-real":avgs[4],"std-real-real":avgs[5]})
	avgframe.to_csv(datapath + "simdiffavgs.csv")

	plt.plot((0,1),(0,1),c='k',linewidth=0.5)
	plt.errorbar(avgs[0],avgs[1],yerr=avgs[1+3]*2.0,fmt='o',c='b',ecolor='b')
	plt.errorbar(avgs[0],avgs[2],yerr=avgs[2+3]*2.0,fmt='o',c='r',ecolor='r')
	plt.show()


if __name__ == "__main__":
	runit();