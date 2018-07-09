import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

size = 10
grid = 1.0
gridsize = np.int32(np.ceil(size/grid))




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
		disc += [[x,y,d]]
	return points, disc;

mat = np.zeros((gridsize,gridsize))
for i in range(10):
	
	r = np.random.rand(2,2) * size
	# r[1][0] = r[0][0]
	# print(r)
	p,d = splitline(*r[0],*r[1],grid)
	print(p,d)
	# plt.scatter(p.T[0],p.T[1],s=20,c=np.linspace(0.0,1.0,len(p.T[0])))
	plt.plot(r.T[0],r.T[1])
	for j in d:
		# print(j)
		mat[int(j[0]),int(j[1])] += j[2]

# plt.pcolormesh(np.flipud(np.fliplr(mat)))
plt.pcolormesh(mat.T)
plt.xticks(np.arange(0,size,grid))
plt.yticks(np.arange(0,size,grid))
plt.grid()
plt.xlim(0,size)
plt.ylim(0,size)
plt.axes().set_aspect(1.0)
plt.show()