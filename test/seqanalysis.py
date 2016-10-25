import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;
import os, sys;
import multiprocessing as mp;
import mkl
import pickle;
import uuid;


sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/test")
import sequences as sq;


def SubTree(start, end, tablepath, groups):
	table = pd.read_csv(tablepath)
	g = table.groupby(['case']);
	probtree = sq.Tree();
		
	for i in range(start, end):
		i_seq = list(g.get_group(groups[i])['code']);
		for j in range(i+1, len(groups)):
			if(i == j): continue;
			j_seq = list(g.get_group(groups[j])['code']);
			i_out, j_out = sq.Seqcompare(i_seq,j_seq);
			if(len(i_out) > 4):
				sq.TreeAdd(probtree,i_out)
			if(len(j_out) > 4):
				sq.TreeAdd(probtree,j_out)
	return probtree;


if __name__ == '__main__':
	mp.set_start_method('forkserver')
	
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"
	
	smallpath = "/tmp/seq"+ str(uuid.uuid4())
	
	print("reading...")
	acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")

	#reduce mem footprint
	small = pd.DataFrame();
	small['case'] = acttable['TUCASEID']
	small['code'] = acttable['TRCODE']
	small.to_csv(smallpath);
	
	g = small.groupby(['case']);
	groups = list(g.groups);
	count = len(groups);
	
# 	print(count);
	
	print("splitting and processing...")
	seinds = []
	threads = mkl.get_max_threads();
	#threads = 4
	totals = np.floor((count*count - count) / 2);
	fraction = totals/threads;
	seinds += [0];

	val = 0; ind = 1;
	for i in range(0, count):
		val += (i+1);
		if val > fraction:
			seinds += [i];
			ind += 1;
			val = 0;
	seinds += [count];
	seinds.reverse()
	seinds = [count - b for b in seinds]
# 	print(seinds)

	args = []
	for i in range(0, threads):
		args += [(seinds[i],seinds[i+1],smallpath,groups)]
		
	p = mp.Pool(threads);
	trees = p.starmap(SubTree, args);
	
	result = sq.CollapseTrees(trees);

	file = open(datapath + "sequencetree.pickle", 'wb');
	
	pickle.dump(result,file)
	
	file.close()
	
	os.remove(smallpath);


