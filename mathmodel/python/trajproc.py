import numpy as np;
import pandas as pd;
import scipy.sparse;
import sqlite3;
import matplotlib.pyplot as plt;
import sys;
import pyproj;
import datetime;


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


	

print("Start:",datetime.datetime.now().time().isoformat());
con = sqlite3.connect(sys.argv[1]);

maxrow = int(pd.read_sql_query("select count(*) from acttraj", con).iloc[0,0]);
print("Maxrows:",maxrow);
maxrow = 10000000

blocksize = 1000; 
gridsize = 1000.0; # meters
timesplit = 15.0 # minutes


regmaxlat = 41.99773652
regminlat = 38.70051584
regmaxlong = -114.05593872
regminlong = -109.00222778

slots = int(np.floor(1440.0 / timesplit))

print("Infoget:", datetime.datetime.now().time().isoformat());
maxx = pd.read_sql_query("select max(long) from acttraj where long <= " + str(regmaxlong), con).iloc[0,0];
maxy = pd.read_sql_query("select max(lat) from acttraj where lat <= " + str(regmaxlat), con).iloc[0,0];
minx = pd.read_sql_query("select min(long) from acttraj where long >= " + str(regminlong), con).iloc[0,0];
miny = pd.read_sql_query("select min(lat) from acttraj where lat >= " + str(regminlat), con).iloc[0,0];
print("Matpop:", datetime.datetime.now().time().isoformat());
print(minx,miny,maxx,maxy)

mindim = reversetransraw(minx,miny);
maxdim = reversetransraw(maxx,maxy);
lendim = ( np.abs(maxdim[0]-mindim[0]), np.abs(maxdim[1]-mindim[1]));


xd,yd = getgrid(*lendim,gridsize,mindim);

print("Gridsize: ",xd,yd)
mat = scipy.sparse.dok_matrix((xd,yd));

# exit()

for i in range(0, maxrow, blocksize):
	if( i % (100000) == 0): 
		print(i,end=' ');
		sys.stdout.flush();
	d = pd.read_sql_query("select * from acttraj limit "+str(i)+","+str(blocksize),con);
	# print(d);
	for ind,f in d.iterrows():
		end = f.start+f.length
		if( f.start < 655.0 and end > 640.0):
			locx,locy = reversetransraw(f.long,f.lat);
			x,y = getgrid(locx,locy,gridsize,mindim);
			if(x >= 0 and x < xd and y >=0 and y < yd):
				mat[x,y] += 1.0;

con.close();
print("Post build + Plotting:",datetime.datetime.now().time().isoformat());

mat = np.array(mat.todense())
maxval = np.max(mat)
mat = np.clip(mat, 0.0,maxval);
print(maxval, np.argmax(mat)//yd, np.argmax(mat) % yd)
ax = plt.subplot(1,1,1)
ax.set_aspect(1.0)
ax.pcolormesh(mat,vmin=0.0,vmax=maxval,cmap=plt.get_cmap('viridis'));
print("Done:",datetime.datetime.now().time().isoformat());
plt.show();
