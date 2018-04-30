import pandas as pd;
import sqlite3;
import numpy as np;
import sys;
import multiprocessing as mp;
import mkl;
import datetime;

blocksize = 1600;
chunksplit = 100;
path = sys.argv[1]
initcol = ["start","lat","long"]


def fixer(frames):
	
	outcon = sqlite3.connect(path +"-"+str(mp.current_process().name)+ ".fix");

	for ind,gdf in frames:
	
		swapcol = []
		for c in initcol:
			# m = gdf[c].min()
			a = gdf[c].mean()
			# print(m,a)
			if( np.abs(a+111.0) < 5.0):
				swapcol += ["long"]
			elif( np.abs(a-40.0) < 5.0):
				swapcol += ["lat"]
			else:
				swapcol += ["start"]
		# print(swapcol)
		gdf.rename(columns={initcol[0]:swapcol[0],initcol[1]:swapcol[1],initcol[2]:swapcol[2]},inplace=True)

		gdf.to_sql('acttraj',outcon,if_exists='append');

	outcon.close();




def main(threads):

	print("Start:",datetime.datetime.now().time().isoformat());

	con = sqlite3.connect(path);

	maxagent = int(pd.read_sql_query("select max(agentnum) from acttraj", con).iloc[0,0]);

	p = mp.Pool(threads);

	for i in range(0,maxagent, blocksize):
	# for i in range(2364000,2364001, blocksize):
		print(i,end=' ',flush=True);


		df = pd.read_sql_query("select * from acttraj where agentnum >= "+str(i)+" and agentnum < "+str(i+blocksize), con);

		g = list(df.groupby("agentnum"))

		splittable = [];
		for i in range(0,len(g),chunksplit):
			splittable += [g[i:min(len(g),i+chunksplit)]]
		p.map(fixer,splittable);

	p.close();
	con.close();

	print("")
	print("Finish:",datetime.datetime.now().time().isoformat());

if __name__ == '__main__':
	threads = mkl.get_max_threads();
	# threads = 2;
	main(threads)
