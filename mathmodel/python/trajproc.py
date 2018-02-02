import numpy as np;
import pandas as pd;
import scipy.sparse;
import sqlite3;
import matplotlib.pyplot as plt;
import sys;
import pyproj;
import datetime;
import h5py;


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

	

print("Start:",datetime.datetime.now().time().isoformat());
con = sqlite3.connect(sys.argv[1]);

maxrow = int(pd.read_sql_query("select count(1) from acttraj", con).iloc[0,0]);
print("Maxrows:",maxrow);
# maxrow = 100000

blocksize = 100000; 
gridsize = 100.0; # meters
timesplit = 15.0 # minutes


regmaxlat = 41.99773652
regminlat = 38.70051584
regminlong = -114.05593872
regmaxlong = -109.00222778

slots = int(np.floor(1440.0 / timesplit))

# print("Infoget:", datetime.datetime.now().time().isoformat());
# maxx = pd.read_sql_query("select max(long) from acttraj where long <= " + str(regmaxlong), con).iloc[0,0];
# maxy = pd.read_sql_query("select max(lat) from acttraj where lat <= " + str(regmaxlat), con).iloc[0,0];
# minx = pd.read_sql_query("select min(long) from acttraj where long >= " + str(regminlong), con).iloc[0,0];
# miny = pd.read_sql_query("select min(lat) from acttraj where lat >= " + str(regminlat), con).iloc[0,0];
print("Matpop:", datetime.datetime.now().time().isoformat());
# print(minx,miny,maxx,maxy)


mindim = reversetransraw(regminlong,regminlat);
maxdim = reversetransraw(regmaxlong,regmaxlat);
lendim = ( np.abs(maxdim[0]-mindim[0]), np.abs(maxdim[1]-mindim[1]));


xd,yd = getgrid(*lendim,gridsize,(0.0,0.0));

print("Gridsize: ",xd,yd)
mat = scipy.sparse.dok_matrix((xd,yd));

# exit()

lastact = [-1.,-1.,-1.,-1.] #act, agentnum, lastx, lasty

winmax = 655.0
winmin = 640.0

for i in range(0, maxrow, blocksize):
	if( i % (100000) == 0): 
		print(i,end=' ');
		sys.stdout.flush();
	d = pd.read_sql_query("select * from acttraj limit "+str(i)+","+str(blocksize),con);
	# print(d);
	for ind,f in d.iterrows():
		end = f.start+f.length
		if( f.start < winmax and end > winmin):
			if(f.long <= regmaxlong and f.long >= regminlong and \
			   f.lat <= regmaxlat and f.lat >= regminlat):
				locx,locy = reversetransraw(f.long,f.lat);
				x,y = getgrid(locx,locy,gridsize,mindim);
				# if(x >= 0 and x < xd and y >=0 and y < yd):
				weight = min(end, winmax) - max(f.start,winmin)
				if(lastact[0] == f.actcode and lastact[1] == f.agentnum):
					line = bresenham(x,y,lastact[2],lastact[3])[:-1];
					cline = len(line)
					for t in range(len(line)):
						mat[line[t][0],line[t][1]] += (weight / cline);
				else:
					mat[x,y] += weight;
				lastact = [f.actcode,f.agentnum,x,y]
				
con.close();
print("Post build + Plotting:",datetime.datetime.now().time().isoformat());

mat = np.array(mat.todense())


outfile = h5py.File(sys.argv[1]+".h5");
outfile.create_dataset("/testtraj",data=mat,fillvalue=0.,compression='gzip',compression_opts=9)
outfile.close();


# mat = np.log10(mat + 1)
maxval = np.max(mat)
maxval = 0.1;
mat = np.clip(mat, 0.0,maxval);

print(maxval, np.argmax(mat)//yd, np.argmax(mat) % yd)
ax = plt.subplot(1,1,1)
ax.set_aspect(1.0)
ax.pcolormesh(mat,vmin=0.0,vmax=maxval,cmap=plt.get_cmap('viridis'));
print("Done:",datetime.datetime.now().time().isoformat());
plt.show();
