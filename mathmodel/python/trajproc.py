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
import mkl;
import sqlalchemy;

blocksize = 1600; #agent records
chunksplit = 50; 
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

def altgetgrid(locx,locy, grid, mindims):
	x = (locx-mindims[0]) / grid
	y = (locy-mindims[1]) / grid
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

def dist(x0,y0,x1,y1):
	dx = x1-x0; dy = y1-y0;
	return np.sqrt(dx*dx+dy*dy)

def splitline(x0,y0,x1,y1,grid=1.0):
	if(x0 == x1 and y0 == y1):
		return [[x0,y0,1.0]]

	flipflag = False;
	if(x0 == x1):
		flipflag = True
		x0,y0,x1,y1 = y0,x0,y1,x1

	points = [[x0,y0],[x1,y1]];
	# print(points)
	dx = x1-x0
	dy = y1-y0

	minx = np.minimum(x0,x1)
	miny = np.minimum(y0,y1)
	maxx = np.maximum(x0,x1)
	maxy = np.maximum(y0,y1)
	
	offx = grid - np.fmod(minx,grid)
	offy = grid - np.fmod(miny,grid)

	m = dy/dx
	b = y0 - m*x0
	
	for x in np.arange(minx+offx,maxx,grid):
		points += [[x,m*x+b ]]

	for y in np.arange(miny+offy,maxy,grid):
		points += [[ (y - b)/m,y]]

	points = np.array(points)
	plist = np.lexsort((points.T[0],points.T[1]))
	points = points[plist]

	if flipflag:
		points = points.T
		flip = np.array([points[1],points[0]])
		points = flip.T

	disc = []
	tdist = dist(x0,y0,x1,y1)
	for i in range(1,len(points)):
		d = dist(points[i][0],points[i][1],points[i-1][0],points[i-1][1])
		x = np.floor((points[i-1][0] +  points[i][0]) / 2)
		y = np.floor((points[i-1][1] +  points[i][1]) / 2)
		disc += [[int(x),int(y),d/tdist]]
	return disc;

def bsrify(mat):
	newmat = []
	for i in range(len(mat)):
		submat = []
		for j in range(len(mat[i])):
			submat += [mat[i][j].tobsr()]
		newmat += [submat]
	return newmat;

def wmax(mat):
	maxvals = np.zeros((len(mat[0])))
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			maxvals[j] = max(mat[i][j].max(),maxvals[j])
	return maxvals;

def speedlimit(x0,y0,x1,y1,t,limit):
	if not (t > 0): return True;
	dx,dy = x1-x0,y1-y0
	d = np.sqrt(dx*dx + dy*dy)
	if(d/t > limit): return True;
	return False;


def runblock(frames, griddims, stepsize):

	# print(frame.iloc[0]['agentnum'],end = "A ", flush=True)
	steps = 1440 // stepsize;
	
	# engine = sqlalchemy.create_engine("sqlite:///"+path);
	# con = sqlite3.connect(path);


	xd,yd,gs,mindim = griddims;

	mats = [];
	
	for i in range(0, int(steps)):
		#res,work,trans
		mats += [[scipy.sparse.dok_matrix((xd,yd)), \
			scipy.sparse.dok_matrix((xd,yd)), \
			scipy.sparse.dok_matrix((xd,yd))]];

	# print("Lenmats ",len(mats))
	# print("Dims ", griddims)

	

	winmin = np.arange(0,1440,stepsize)
	winmax = winmin + stepsize
	# print(winmin, winmax)

	# for i in range(st, en, blocksize):
	# 	if( i % (blocksize) == 0): 
	# 		print(i,end=' ');
	# 		sys.stdout.flush();
	# 	outblock = (en - i) if( (en - i) < blocksize ) else blocksize;
		
		# d = pd.read_sql_query("select * from acttraj limit "+str(i)+","+str(outblock),con);
	
	# print(frame.iloc[0].long,frame.iloc[0].lat)


	# x,y = 0,0
	# if(frame.iloc[0].long <= regmaxlong and frame.iloc[0].long >= regminlong and \
	# frame.iloc[0].lat <= regmaxlat and frame.iloc[0].lat >= regminlat):
	# 	locx,locy = reversetransraw(frame.iloc[0].long,frame.iloc[0].lat);
	# 	x,y = getgrid(locx,locy,gs,mindim);

	# lastact = [frame.iloc[0].actcode,frame.iloc[0].actcode,x,y]
	for ff,frame in frames:
		frame[['locx','locy']] = frame[['long','lat']]
		frame[['locx','locy']] = frame[['locx','locy']].apply(reversetrans,axis=1)
		frame['x'] = frame['locx'].apply(lambda x: (x - mindim[0])/gs)
		frame['y'] = frame['locy'].apply(lambda x: (x - mindim[1])/gs)
		frame['nx'] = frame['x'].shift(-1)
		frame['ny'] = frame['y'].shift(-1)

			# print(d);
		
		for ind,f in frame.iterrows():
			start = max(0,f.start)
			end = min(f.start+f.length,1439)
			#if( f.start < winmax and end > winmin):
			slotst = int(np.floor(start/stepsize))
			sloten = int(np.floor(end/stepsize))+1
			# print(f.actcode,f.start,slotst,end,sloten)
			# try:
			for slot in range(int(slotst),int(sloten)):
				
				# winmin = slot*stepsize; winmax = winmin + stepsize;
				# if(f.long <= regmaxlong and f.long >= regminlong and \
				# f.lat <= regmaxlat and f.lat >= regminlat):
				if( (f.x >= 0.0)& (f.y >= 0.0) & (f.x < xd) & (f.y < yd) & (f.nx < xd) & (f.ny < yd)):
					# locx,locy = reversetransraw(f.long,f.lat);
					# x,y = getgrid(locx,locy,gs,mindim);
					# x,y = altgetgrid(locx,locy,gs,mindim);
					# if(x >= 0 and x < xd and y >=0 and y < yd):
					weight = min(end, winmax[slot]) - max(f.start,winmin[slot])
					# print(weight)
					if(f.actcode >= 180000 and f.actcode <=189999):
						if(np.isnan(f.nx) or np.isnan(f.ny)):
							mats[slot][2][int(np.floor(f.x)),int(np.floor(f.y))] += weight
							continue;
						# if(lastact[0] == f.actcode and lastact[1] == f.agentnum):
							#speed limit - we assume max speed of 50 m/s 
						# if not(speedlimit(f.x,f.y,f.nx,f.ny,weight, 50.0)):
							# mats[slot][2][x,y] += weight;
						# else:
							# line = bresenham(f.x,f.y,f.nx,f.ny);
						line = splitline(f.x,f.y,f.nx,f.ny)
						# if((f['y'] < 1939) & (f['y'] > 1932) & (f['x'] > 1890) & (f['x'] < 1895)):
							# print(line)
						# cline = len(line)
						for t in range(len(line)):
							mats[slot][2][line[t][0],line[t][1]] += weight * line[t][2];
						# else:
							# mats[slot][2][x,y] += weight;
					elif (f.actcode >= 50000 and f.actcode <=59999):
						mats[slot][1][int(np.floor(f.x)),int(np.floor(f.y))] += weight;
					else: 
						mats[slot][0][int(np.floor(f.x)),int(np.floor(f.y))] += weight;
			# except IndexError:
				# print(f.agentnum,ind,slot,f.x,f.y,f.nx,f.ny);
						
	# con.close();

	return bsrify(mats);



def chunker(path,q, st,en):
	
	con = sqlite3.connect(path);

	for i in range(st, en, blocksize):
		if( (i % blocksize) == 0): 
			print(i,end='L ');
			sys.stdout.flush();
		# outblock = (en - i) if( (en - i) < blocksize ) else blocksize;
		#outblock + 1 because we need overlap between frames for accuracy
		df = pd.read_sql_query("select * from acttraj where agentnum >= "+str(i)+" and agentnum < "+str(i+blocksize), con);
		q.put(df);

	con.close();

def dosplit(g,threads,griddims,stepsize):
	splittable = [];

	# size = len(df) - 1

	# splitsize = (size // threads) + 1

	# st = 0;


	return splittable;


def matsum(matlist):
	#implicitly, all the mats 
	out = matlist[0]

	for i in range(1,len(matlist)):
		#bsrdmat = bsrify(out[i]);
		for j in range(len(matlist[i])):
			for k in range(len(matlist[i][j])):
				out[j][k] += matlist[i][j][k]

	return out;

def parallelrunblock(threads, path, st, en, griddims, stepsize):

	q = mp.Queue(5);
	ch = mp.Process(target=chunker, args=(path,q,st,en));
	ch.start();

	p = mp.Pool(threads);

	xd,yd,gs,mindim = griddims;

	mats = [];
	steps = 1440 // stepsize;
	for i in range(0, int(steps)):
		#res,work,trans
		mats += [[scipy.sparse.bsr_matrix((xd,yd)), \
			scipy.sparse.bsr_matrix((xd,yd)), \
			scipy.sparse.bsr_matrix((xd,yd))]];


	for v in range(st,en,blocksize):
		# if( (v % blocksize) == 0): 
		print(v,end='R ');
		sys.stdout.flush();

		df = q.get();
		g = list(df.groupby("agentnum"))
		# splittable = dosplit(df,threads - 1, griddims, stepsize)
		# print(g,flush=True)
		splittable = [];
		for i in range(0,len(g),chunksplit):
			splittable += [(g[i:min(len(g),i+chunksplit)],griddims,stepsize)]
		# st += splitsize;

		# print(splittable,flush=True)
		out = p.starmap(runblock,splittable);
		
		# print(len(out)," ",v,end='pre ',flush='true');

		# splittable = []
		# for i in range(0,len(out),100):
		# 	splittable += [ out[i,min(i+100,len(out))] ]
		# out = p.map(matsum, splittable)

		# print(v,end='post ',flush='true');

		out = matsum(out);

		mats = matsum([out,mats]);
		# 
		# print(len(g))
		# out = p.map(processtraj,g,chunksize=100)



		
		# print(out)



	# print(mats)
	# p.close()
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
			submat = np.array(mats[i][j].todense()).T
			
			# outfile.create_dataset("/traj-slot-"+str(i).zfill(3)+"-set-"+str(j).zfill(3),data=submat,fillvalue=0.,compression='gzip',compression_opts=9)
			
			# submat = np.log10(submat + 1)
			submat[submat > 0.0] += (vmax[j] - submat[submat > 0.0])*0.2
			# maxval = np.max(mat)
			# maxval = 0.1;
			# submat = np.clip(submat, 0.0,0.01);
			ax = plt.axes([0.,0.,1.,1.])
			ax.set_axis_off()
			ax.set_aspect(1.0)

			ax.set_ylim(1400,3000)
			ax.set_xlim(1000,2600)
			# print(maxval, np.argmax(mat)//yd, np.argmax(mat) % yd)
			# ax = plt.subplot(i+1,j+1,xpl*ypl)
			# ax.pcolormesh(submat,vmin=0.0,vmax=maxvals[j],cmap=plt.get_cmap('viridis'));
			
			ax.pcolormesh(submat,vmin=0.0,vmax=vmax[j],cmap=plt.get_cmap('viridis'));

			F = plt.gcf();
			F.set_size_inches(8,8) #1600x1600
			F.set_dpi(200.0);
			F.savefig(str(path)+"slot-"+str(i+startind).zfill(2)+"-set-"+str(j).zfill(2)+".png",dpi=300);
			plt.clf()
	# outfile.close();


def parallelplotmats(threads,mats,maxvals,path,nzscale=0.2):


	outfile = h5py.File(path+".h5");

	for i in range(len(mats)):
		for j in range(len(mats[i])):
			submat = np.array(mats[i][j].todense())
			ds = outfile.create_dataset("/traj-slot-"+str(i).zfill(3)+"-set-"+str(j).zfill(3),data=submat,fillvalue=0.,compression='gzip',compression_opts=9)
			ds.attrs['grid'] = gridsize
			ds.attrs['time'] = i * timesplit
			ds.attrs['length'] = timesplit
			ds.attrs['timeunit'] = "minutes"
			ds.flush();
	outfile.close();

	splittable = [];
	size = len(mats)
	split = size // threads;
	# print(split)


	for i in range(threads):
		splittable += [(mats[i*split:(i+1)*split],maxvals,path,i*split)]

	# print(splittable)

	p = mp.Pool(threads);
	out = p.starmap(plotmats,splittable);

def main(threads):

	# threads = 8;

	print("Start:",datetime.datetime.now().time().isoformat());

	path = sys.argv[1]

	con = sqlite3.connect(path);

	# maxrow = int(pd.read_sql_query("select count(1) from acttraj", con).iloc[0,0]);
	# print("Maxrows:",maxrow);
	maxagent = int(pd.read_sql_query("select max(agentnum) from acttraj", con).iloc[0,0]);
	# maxagent = 100
	print(maxagent)
	con.close();

	print("Matpop:", datetime.datetime.now().time().isoformat());
	# print(minx,miny,maxx,maxy)


	# splitsize = maxrow // threads;

	mindim = reversetransraw(regminlong,regminlat);
	maxdim = reversetransraw(regmaxlong,regmaxlat);
	lendim = ( np.abs(maxdim[0]-mindim[0]), np.abs(maxdim[1]-mindim[1]));

	# print(mindim,maxdim,lendim)

	xd,yd = getgrid(*lendim,gridsize,(0.0,0.0));
	griddims = (xd+1,yd+1,gridsize,mindim)
	print("Gridsize: ",griddims)

	# mats = runblock(path,0,maxrow,griddims,timesplit);
	mats = parallelrunblock(threads, path,0, maxagent ,griddims,timesplit);


	print("")
	print("Post build + Plotting:",datetime.datetime.now().time().isoformat());

	# exit();

	# mats = mergemats(mats)
	# mats,maxvals = bsrifywmax(mats);
	maxvals = wmax(mats);

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
	parallelplotmats(threads,mats,maxvals,path)

	print("Done:",datetime.datetime.now().time().isoformat());
	

if __name__ == "__main__":
	threads = mkl.get_max_threads();
	# threads = 8;
	main(threads)