'''
@author Ben
In this example we will compare Shapely AGAINST RTREE for performance
NOTE: to get this to work I had to install libspatialindex 1.5 from source
and then:
export LD_LIBRARY_PATH=/usr/local/lib
'''

import random, os, sys, time, numpy

import rtree;
from rtree import Rtree

# instantiate the Rtree class
p=rtree.index.Property(variant=rtree.index.RT_Star)
RtreeIndex = Rtree(properties=p)


import random, os, sys, time, numpy
from shapely import *
from shapely.geometry import Point

filename = 'rtree_radius_search.png'
numberOfPoints = 500
gridWidth = 10000
gridHeight = 10000
shapePoints = []



circleRadius = random.randint(500, 1500)
circleX = random.randint(0, gridWidth)
circleY = random.randint(0, gridHeight)

boxLeft = circleX - circleRadius
boxRight = circleX + circleRadius
boxTop = circleY + circleRadius
boxBottom = circleY - circleRadius

sCircle = Point(circleX, circleY)
sCircle = sCircle.buffer(circleRadius, 16)

shapePoints = []
pointList = []
for i in range(numberOfPoints):
	x = random.randint(0, gridWidth)
	y = random.randint(0, gridHeight)
	# Add the points to a python list of lists
	pointList.append((x, y))
	# Create a Shapely point object
	iPoint = Point(x, y)
	iPoint.idx = i # set our custom attribute. (if this doesnt work I have other ways)
	shapePoints.append(iPoint)
	
	# Add the point to an RTREE index
	# index.add(id=id, (left, bottom, right, top))
	# Note that this would work if the 
	RtreeIndex.add(i, (x, y, x, y))

matchingPoints = []
searchBench = time.time()
for idx, point in enumerate(shapePoints):
	if sCircle.contains(point):
		matchingPoints.append(idx)

searchBench = time.time() - searchBench

#-------------------------------------------------------------------------------
# NOW TEST RTREE
RtreeBench = time.time()
RtreeIndex
rtreeMatches = list(RtreeIndex.intersection((boxLeft, boxBottom, boxRight, boxTop)))

RtreeBench = time.time() - RtreeBench

print("\n\n")
print('SHAPELY: There were {:d} points within the circle [{:d},{:d}] - r[{:d}]'.format(len(matchingPoints), circleX, circleY, circleRadius))
print('SHAPELY: Calculation Took {} seconds for {} points'.format(searchBench, numberOfPoints))
print( '------------------------------------------------------------------')
print( 'RTREE: There were {:d} points within the box [x1:{:d},y1:{:d}, x2:{:d}, y2:{:d}]'.format(len(rtreeMatches), boxLeft, boxBottom, boxRight, boxTop))
print( 'SHAPELY: Calculation Took {} seconds for {} points\n'.format(RtreeBench, numberOfPoints))
print( 'Saving Graph to {}'.format(filename))
print( '\n\n')


#_-------------------------------------------------------------------------------
# DRAW A REPRESENTATION OF THE LOGIC ABOVE:
import matplotlib
matplotlib.use('Agg') # Do NOT attempt to open X11 instance
from pylab import *
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as pyplot


matplotlib.rcParams['lines.linewidth'] = 2


pyplot.axis([0, gridWidth, 0, gridHeight])
# Setting the axis labels.
pyplot.xlabel('X Space')
pyplot.ylabel('Y Space')

#Give the plot a title
pyplot.title('Radius Search Plot Using Shapely')

# Draw the collision circle/boundary
cir = Circle((circleX, circleY), radius=circleRadius, fc='b',zorder=2)
cir.set_alpha(0.1)

x = []
y = []
col = []

clist=['c','m','y','k','b','g','r']

b = RtreeIndex.leaves();
for i in b:
	pyplot.gca().add_patch(Rectangle((i[2][0],i[2][1]),i[2][2]-i[2][0],i[2][3]-i[2][1],fill=False))

for idx, point in enumerate(pointList):
	style = 'g' 
	#iAlpha = 0.4
	if(idx in matchingPoints):
		style = 'r'
	#	iAlpha = 1
	x += [point[0]]
	y += [point[1]]
	col += [style]
	#pyplot.plot(point[0], point[1], style, linewidth=1, markersize=3, alpha=iAlpha)
pyplot.scatter(x,y,c=col,s=10,marker='o',alpha=0.4,zorder=3)

pyplot.gca().add_patch(cir)
pyplot.gca().set_aspect(1.0)
pyplot.savefig(os.getcwd()+'/'+str(filename), dpi=200)

