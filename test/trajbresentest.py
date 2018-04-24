
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import sqlite3
import pyproj;


regmaxlat = 41.99773652
regminlat = 38.70051584
regminlong = -114.05593872
regmaxlong = -109.00222778
grid = 100.0; # meters

outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');

def latlongtrans(x):
	x.locx,x.locy=pyproj.transform(inproj,outproj,x.locx,x.locy);
	return x

def reversetrans(x):
	x.long,x.lat=pyproj.transform(outproj,inproj,x.long,x.lat);
	return x

def latlongtransraw(locx,locy):
	locx,locy=pyproj.transform(inproj,outproj,locx,locy);
	return locx,locy

def reversetransraw(locx,locy):
	locx,locy=pyproj.transform(outproj,inproj,locx,locy);
	return locx,locy


def bresenham(x0,y0,x1,y1):
	dx = x1-x0
	dy = y1-y0
	sx = np.int32(np.sign(dx))
	sy = np.int32(np.sign(dy))
	dx = np.int32(np.abs(dx));
	dy = np.int32(np.abs(dy));
	err = dx - dy
	# print(err,type(err))
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

def splitline(x0,y0,x1,y1,grid):

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
		disc += [[x,y,d/tdist]]
	return  np.array(disc);

mindim = reversetransraw(regminlong,regminlat);
maxdim = reversetransraw(regmaxlong,regmaxlat);
lendim = np.array( [np.abs(maxdim[0]-mindim[0]), np.abs(maxdim[1]-mindim[1])] );
print(lendim)
mat = np.zeros(np.int32(np.ceil(lendim/grid)))
# mat[1900,1900] = 1.0

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
con = sqlite3.connect(datapath + "Ftraj4-2018-04-19_16-53-21-1.merge.sqlite3")
df = pd.read_sql_query("select * from acttraj limit 10000",con)

df[['x','y']] = df[['long','lat']].apply(reversetrans,axis=1)

df['x'] = df['x'].apply(lambda x: (x - mindim[0])/grid)
df['y'] = df['y'].apply(lambda x: (x - mindim[1])/grid)


# df['xl'] = df['x'].apply(lambda x: x/grid) 
# df['yl'] = df['y'].apply(lambda x: x/grid)

# df['xl'] = df['x'].apply(lambda x: (x - mindim[0])/grid) 
# df['yl'] = df['y'].apply(lambda x: (x - mindim[1])/grid)

# df['x'] = df['xl'].apply(lambda x: np.int32(np.floor(x)))
# df['y'] = df['yl'].apply(lambda x: np.int32(np.floor(x)))
df['nx'] = df['x'].shift(-1)
df['ny'] = df['y'].shift(-1)

g = df.groupby("agentnum")



for ind,gf in g:
	print(gf[(gf['y'] < 1952) & (gf['y'] > 1938) & (gf['x'] > 1892) & (gf['x'] < 1906)][['x','y','length']])
	for ind2,gfr in gf[:-1].iterrows():
		# print(gfr)
		# if(gfr['y'] < 1952 and gfr['y'] > 1938 and gfr['x'] > 1892 and gfr['x'] < 1906):
			# print(gfr)
		if(gfr.length <= 0):
			l = 0.001
		else:
			l = gfr.length
		if(gfr.x == gfr.nx and gfr.y == gfr.ny):
			mat[int(np.floor(gfr.x)),int(np.floor(gfr.y))] += l
		else:
			# line = bresenham(gfr.x,gfr.y,gfr.nx,gfr.ny)
			line = splitline(gfr.x,gfr.y,gfr.nx,gfr.ny,1.0)
			for i in line:
				# print(i)
				# print(i)
				mat[int(i[0]),int(i[1])] += l * i[2]

matmax = np.max(mat)
print(matmax)

plt.pcolormesh(mat.T,vmax=1.0)

for ind,gf in g:
	plt.plot(gf['x'],gf['y'],linewidth=0.5)
	plt.scatter(gf['x'],gf['y'],s=gf['length']*5)


plt.axes().set_aspect(1.0)
plt.show()

