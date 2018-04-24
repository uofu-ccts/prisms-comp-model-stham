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
gridsize = 100.0
stepsize=15
regmaxlat = 41.99773652
regminlat = 38.70051584
regminlong = -114.05593872
regmaxlong = -109.00222778


outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');

def latlongtrans(x):
	x.locx,x.locy=pyproj.transform(inproj,outproj,x.locx,x.locy);
	return x

def reversetrans(x):
	x.locx,x.locy=pyproj.transform(outproj,inproj,x.locx,x.locy);
	return x

def latlongtransraw(locx,locy):
	locx,locy=pyproj.transform(inproj,outproj,locx,locy);
	return locx,locy

def reversetransraw(locx,locy):
	locx,locy=pyproj.transform(outproj,inproj,locx,locy);
	return locx,locy

def getgrid(locx,locy, grid, mindims):
	x = int(np.floor(np.abs(locx-mindims[0]) / grid))
	y = int(np.floor(np.abs(locy-mindims[1]) / grid))
	return x,y

mindim = reversetransraw(regminlong,regminlat);
maxdim = reversetransraw(regmaxlong,regmaxlat);


def bresenham(x0,y0,x1,y1):
	dx = x1-x0
	dy = y1-y0
	sx = np.sign(dx)
	sy = np.sign(dy)
	dx = np.abs(dx);
	dy = np.abs(dy)
	err = dx - dy

	out = [(x0,y0)]
	x = x0;
	y = y0;

	while(x != x1 or y != y1):
		e2 = err << 1
		if(e2 > -dy):
			err -= dy
			x += sx
		if(e2 < dx):
			err += dx
			y += sy
		out += [(x,y)]
	return out;

def speedlimit(x0,y0,x1,y1,t,limit):
	if not (t > 0): return True;
	dx,dy = x1-x0,y1-y0
	d = np.sqrt(dx*dx + dy*dy)
	if(d/t > limit): return True;
	return False;

def processtraj(gframe):

	steps = 1440 // stepsize;

	exp = np.zeros(steps);

	ind,frame = gframe;

	winmin = np.arange(0,1440,stepsize)
	winmax = winmin + stepsize

	x,y = 0,0
	if(frame.iloc[0].long <= regmaxlong and frame.iloc[0].long >= regminlong and \
	frame.iloc[0].lat <= regmaxlat and frame.iloc[0].lat >= regminlat):
		locx,locy = reversetransraw(frame.iloc[0].long,frame.iloc[0].lat);
		x,y = getgrid(locx,locy,gridsize,mindim);

	lastact = [frame.iloc[0].actcode,frame.iloc[0].actcode,x,y]

		# print(d);
	for ind,f in frame.iloc[1:].iterrows():
		end = min(f.start+f.length,1439)
		#if( f.start < winmax and end > winmin):
		slotst = int(np.floor(f.start/stepsize))
		sloten = int(np.floor(end/stepsize))+1
		# print(f.actcode,f.start,slotst,end,sloten)
		for slot in range(int(slotst),int(sloten)):
			# winmin = slot*stepsize; winmax = winmin + stepsize;
			if(f.long <= regmaxlong and f.long >= regminlong and \
			f.lat <= regmaxlat and f.lat >= regminlat):
				locx,locy = reversetransraw(f.long,f.lat);
				x,y = getgrid(locx,locy,gridsize,mindim);
				# if(x >= 0 and x < xd and y >=0 and y < yd):

								

				weight = min(end, winmax[slot]) - max(f.start,winmin[slot])


				# print(weight)
				if(f.actcode >= 180000 and f.actcode <=189999):
					if(lastact[0] == f.actcode and lastact[1] == f.agentnum):
						#speed limit - we assume max speed of 50 m/s 
						if not(speedlimit(x,y,lastact[2],lastact[3],weight, 50.0)):
							# mats[slot][2][x,y] += weight;
						# else:
							line = bresenham(x,y,lastact[2],lastact[3]);
							cline = len(line)
							for t in range(len(line)):
								exp[slot] += (weight / cline) * mats[slot][line[t][0],line[t][1]];

				else: 
					exp[slot] += weight * mats[slot][x,y];

				lastact = [f.actcode,f.agentnum,x,y]



	return exp;


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
	for i in range(0,96):
		print(i)
		mats += [matfile["/traj-slot-" + str(i).zfill(3) + "-set-002"][:]]
	matfile.close()

	p = mp.Pool(processes=threads);
	g = df.groupby("agentnum")
	print(len(g))
	out = p.map(processtraj,g,chunksize=100)
	
	p.close();

	arr = np.concatenate(out);
	outfile = h5py.File(datapath + "finalexptraj.h5")
	ds = outfile.create_dataset("/exptraj",data=arr,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.close();

	print("plotting")
	for i in out: 
		plt.plot(i,alpha=0.5,linewidth=0.1,color='k')

	plt.show();



if __name__ == "__main__":
	threads = mkl.get_max_threads();
	threads = 4;
	main(threads)