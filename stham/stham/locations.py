import pyproj
import numpy as np;

#does this global stuff work?
latlongproj = pyproj.Proj(init='epsg:4326');
xyproj = pyproj.Proj(init='epsg:3560');

def latlongtoxy(long,lat):
	locx,locy=pyproj.transform(latlongproj,xyproj,long,lat);
	return locx,locy

def xytolatlong(locx,locy):
	long,lat=pyproj.transform(xyproj,latlongproj,locx,locy);
	return long,lat

def setproj(latlong,xy):
	global latlongproj,xyproj
	latlongproj = pyproj.Proj(init=latlong);
	xyproj = pyproj.Proj(init=xy);


def getgrid(x,y,x0,y0,res):
	x = int(np.floor(np.abs(x-x0) / grid))
	y = int(np.floor(np.abs(y-y0) / grid))
	return x,y


def taxicabtrip(x0,y0,x1,y1):
	#assumes GPS input?
	#returns a time,x,y
	return 


def bresenham(x0,y0,x1,y1):
	"""
	Implements Bresenham's line algorithm. 
	inputs should be integer positions
	returns an numpy array of positions int

	"""
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
	return np.array(out,dtype=np.int);

def dist(x0,y0,x1,y1):
	dx = x1-x0; dy = y1-y0;
	return np.sqrt(dx*dx+dy*dy)

def buildloctree(locs):
	p=rtree.index.Property(variant=rtree.index.RT_Star)
	idx = rtree.index.Rtree(properties=p)
	for i in locs:
		idx.insert(i[0],(i[1],i[2],i[1],i[2]));
	return idx


def loadLocs(file, target, aux, xyproj, meta=None, latlongproj='epsg:4326'):
	"""
	Converts a location input file to an sqlite3 format useable by STHAM
	file - the input file to be converted
	target - the sqlite file that will be produced
	aux - the auxiliary file that contains converted locations
	xyproj - the local coordinate reference system for the region in meters
	meta - a csv containing a description of location types. defaults to None
	latlongproj - the projection of the lat/long coords, defaults to WGS 84

	The file should have column headers 'type', 'lat', and 'long'; order doesn't matter
	'type' should be an integer, 32 or 64 bit depending on need. 

	The output is appended to an sqlite3 table named locs with the same column names
	The auxiliary file also contains a table called locs, with column names 
	x,y, corresponding to the x/y coordinates in the target projection
	and an integer grid. gridding is performed later in definegrid()

	The meta file has three columns: type, alttype, and desc, for the type, an
	optional alternate type id, and a description

	"""
	pass;



def definegrid(x0,y0,x1,y1,aux,grid):
	"""
	Performs the grid instantiation of all locations
	"""

def runit():
	x,y = latlongtoxy(129.7,32.58)
	print(x,y)
	setproj('epsg:4326','epsg:3968')
	x,y = latlongtoxy(129.7,32.58)
	print(x,y)
	long,lat = xytolatlong(x,y)
	print(long,lat)

if __name__ == '__main__':
	runit();