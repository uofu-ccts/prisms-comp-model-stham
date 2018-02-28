import numpy as np;
import pandas as pd;
import scipy.sparse;
import sqlite3;
import matplotlib.pyplot as plt;
import sys;
import pyproj;
import datetime;
import h5py;
import multiprocessing as mp;

blocksize = 100000; #records
gridsize = 100.0; # meters
timesplit = 15.0 # minutes

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

def bsrifywmax(mat):
	newmat = []
	maxvals = np.zeros((len(mat[0])))
	for i in range(len(mat)):
		submat = []
		for j in range(len(mat[i])):
			submat += [mat[i][j].tobsr()]
			maxvals[j] = max(submat[j].max(),maxvals[j])
		newmat += [submat]
	return newmat,maxvals;

def speedlimit(x0,y0,x1,y1,t,limit):
	if not (t > 0): return True;
	dx,dy = x1-x0,y1-y0
	d = np.sqrt(dx*dx + dy*dy)
	if(d/t > limit): return True;
	return False;


def runblock(path, st, en, griddims, stepsize):


	steps = 1440 // stepsize;
	
	con = sqlite3.connect(path);
	
	xd,yd,gs,mindim = griddims;

	mats = [];
	
	for i in range(0, int(steps)):
		#res,work,trans
		mats += [[scipy.sparse.dok_matrix((xd,yd)), \
			scipy.sparse.dok_matrix((xd,yd)), \
			scipy.sparse.dok_matrix((xd,yd))]];

	# print("Lenmats ",len(mats))
	# print("Dims ", griddims)

	lastact = [-1.,-1.,-1.,-1.]

	winmin = np.arange(0,1440,stepsize)
	winmax = winmin + stepsize
	# print(winmin, winmax)

	for i in range(st, en, blocksize):
		if( i % (blocksize) == 0): 
			print(i,end=' ');
			sys.stdout.flush();
		d = pd.read_sql_query("select * from acttraj limit "+str(i)+","+str(blocksize),con);
		# print(d);
		for ind,f in d.iterrows():
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
					x,y = getgrid(locx,locy,gs,mindim);
					# if(x >= 0 and x < xd and y >=0 and y < yd):
					weight = min(end, winmax[slot]) - max(f.start,winmin[slot])
					# print(weight)
					if(f.actcode >= 180000 and f.actcode <=189999):
						if(lastact[0] == f.actcode and lastact[1] == f.agentnum):
							#speed limit - we assume max speed of 50 m/s 
							if not(speedlimit(x,y,lastact[2],lastact[3],weight, 50.0)):
								# mats[slot][2][x,y] += weight;
							# else:
								line = bresenham(x,y,lastact[2],lastact[3])[:-1];
								cline = len(line)
								for t in range(len(line)):
									mats[slot][2][line[t][0],line[t][1]] += (weight / cline);
						# else:
							# mats[slot][2][x,y] += weight;
					elif (f.actcode >= 50000 and f.actcode <=59999):
						mats[slot][1][x,y] += weight;
					else: 
						mats[slot][0][x,y] += weight;

					lastact = [f.actcode,f.agentnum,x,y]
					
	con.close();

	return mats;

def plotmats(mats, maxvals, path, startind, stride=1,st=0,en=-1,nzscale=0.2):

	if(en == -1):
		en = len(mats)

	vmax = np.array(maxvals)
	vmax = np.full_like(vmax, 1)

	# outfile = h5py.File(sys.argv[1]+".h5");
	for i in range(st,en,stride):
		for j in range(len(mats[i])):
			# print(i,j,i//stride)
			submat = np.array(mats[i][j].todense())
			
			# outfile.create_dataset("/traj-slot-"+str(i)+"-set-"+str(j),data=mat,fillvalue=0.,compression='gzip',compression_opts=9)
			
			# submat = np.log10(submat + 1)
			submat[submat > 0.0] += (vmax[j] - submat[submat > 0.0])*0.2
			# maxval = np.max(mat)
			# maxval = 0.1;
			# submat = np.clip(submat, 0.0,0.01);
			ax = plt.subplot(111)
			ax.set_aspect(1.0)

			ax.set_ylim(1300,2800)
			ax.set_xlim(1300,2800)
			# print(maxval, np.argmax(mat)//yd, np.argmax(mat) % yd)
			# ax = plt.subplot(i+1,j+1,xpl*ypl)
			# ax.pcolormesh(submat,vmin=0.0,vmax=maxvals[j],cmap=plt.get_cmap('viridis'));
			ax.pcolormesh(submat,vmin=0.0,vmax=vmax[j],cmap=plt.get_cmap('viridis'));

			F = plt.gcf();
			F.set_size_inches(8,8)
			F.set_dpi(300.0);
			F.savefig(str(path)+"slot-"+str(i+startind)+"-set-"+str(j)+".png",dpi=300);
			plt.clf()


def parallelplotmats(threads,mats,maxvals,path,nzscale=0.2):
	splittable = [];
	size = len(mats)
	split = size // threads;
	# print(split)

	for i in range(threads):
		splittable += [(mats[i*split:(i+1)*split],maxvals,path,i*split)]

	# print(splittable)

	p = mp.Pool(threads);
	out = p.starmap(plotmats,splittable);


def main():

	threads = 2;

	print("Start:",datetime.datetime.now().time().isoformat());

	path = sys.argv[1]

	con = sqlite3.connect(path);

	maxrow = int(pd.read_sql_query("select count(1) from acttraj", con).iloc[0,0]);
	print("Maxrows:",maxrow);
	# maxrow = 200


	con.close();

	print("Matpop:", datetime.datetime.now().time().isoformat());
	# print(minx,miny,maxx,maxy)


	splitsize = maxrow // threads;

	mindim = reversetransraw(regminlong,regminlat);
	maxdim = reversetransraw(regmaxlong,regmaxlat);
	lendim = ( np.abs(maxdim[0]-mindim[0]), np.abs(maxdim[1]-mindim[1]));

	# print(mindim,maxdim,lendim)

	xd,yd = getgrid(*lendim,gridsize,(0.0,0.0));
	griddims = (xd,yd,gridsize,mindim)
	print("Gridsize: ",griddims)

	mats = runblock(path,0,maxrow,griddims,timesplit);

	print("")
	print("Post build + Plotting:",datetime.datetime.now().time().isoformat());

	# mats = mergemats(mats)
	mats,maxvals = bsrifywmax(mats);

	steps = int(np.floor(1440/timesplit))

	stride = 1

	# f,ax = plt.subplots(xpl,ypl,sharex='col', sharey='row')

	# print(xpl,ypl,len(mats),len(ax),len(ax[0]))

	print(maxvals)
	# print(np.log10(maxvals))

			
	# outfile.close();
	
	# f.subplots_adjust(hspace=0.05,wspace=0.05)
	print("Plottime:",datetime.datetime.now().time().isoformat());

	# plt.show();
	# plotmats(mats,maxvals,path,stride=1);
	parallelplotmats(4,mats,maxvals,path)

	print("Done:",datetime.datetime.now().time().isoformat());
	

if __name__ == "__main__":
	main();