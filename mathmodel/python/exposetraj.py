import pandas as pd;
import numpy as np;
import sqlite3;
import mkl;
import multiprocessing as mp;
import sys;
import pyproj;
import h5py;
import matplotlib.pyplot as plt;
import sklearn.ensemble
import datetime
import sys;
import time;

blocksize = 10000

regmaxlat = 41.99773652
regminlat = 38.70051584
regminlong = -114.05593872
regmaxlong = -109.00222778


def getexp(df, slicemat):
	exp = np.zeros(96);



	return exp;




def processtraj(df):

	exptraj = np.zeros(96);
	# print(df[0], " ",len(df[1]))

	return exptraj;


# def chunker(path, q):
# 	con = sqlite3.connect(path);

# 	for i in range(st, en, blocksize):
# 		if( i % (blocksize) == 0): 
# 			print(i,end='L ');
# 			sys.stdout.flush();
# 		outblock = (en - i) if( (en - i) < blocksize ) else blocksize;
# 		#outblock + 1 because we need overlap between frames for accuracy
# 		d = pd.read_sql_query("select * from acttraj limit "+str(i)+","+str(outblock+1),con);

# 		q.put(d);

# 	con.close();


def main(threads):

	# inpath = sys.argv[1];
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	path = datapath + "Ftraj4-2018-03-19_13-53-28-24116.merge.sqlite3"
	con = sqlite3.connect(path);
	
	maxagent = int(pd.read_sql_query("select max(agentnum) from acttraj", con).iloc[0,0]);
	st = 0; en = 10000;
	df = pd.read_sql_query("select * from acttraj where agentnum >= "+str(st)+" and agentnum < "+str(en), con);

	print(len(df))

	con.close();


	exptraj = []

	global mats;
	mats = []
	#96 slots, 3 sets
	matfile = h5py.File(datapath + "Ftraj4-2018-03-19_13-53-28-24116.merge.sqlite3.h5")
	for i in range(0,8):
		print(i)
		mats += [matfile["/traj-slot-" + str(i).zfill(3) + "-set-002"][:]]
	matfile.close()

	p = mp.Pool(processes=threads);
	g = df.groupby("agentnum")
	print(len(g))
	p.map(processtraj,g,chunksize=100)
	
	p.close();


if __name__ == "__main__":
	threads = mkl.get_max_threads();
	threads = 8;
	main(threads)