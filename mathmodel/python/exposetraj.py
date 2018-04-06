import pandas as pd;
import numpy as np;
import sqlite3;
import mkl;
import multiprocessing as mp;
import sysimport;
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

	exptraj = [];

	time.sleep(5)

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

	inpath = sys.argv[1];
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	con = sqlite3.connect(path);
	
	maxagent = int(pd.read_sql_query("select max(agentnum) from acttraj", con).iloc[0,0]);
	st = 0; en = 10000;
	df = pd.read_sql_query("select * from acttraj where agentnum >= "+str(st)+" and < "+str(en));

	con.close();

	exptraj = []

	global mats;
	mats = []
	#96 slots, 3 sets
	matfile = h5py.File(datapath + "Ftraj4-2018-03-19_13-53-28-24116.merge.sqlite3.h5")
	for i in range(0,96):
		mats += [matfile["/traj-slot-" + str(i) + "-set-002"][:]]
	matfile.close()

	p = mp.Pool(processes=threads);

	results = p.map(processtraj,df.groupby("agentnum"),chunksize=100)
	
	


if __name__ == "__main__":
	threads = mkl.get_max_threads();
	threads = 4;
	main(threads)