from osgeo import ogr
import numpy as np
import shapely as sh;
import shapely.wkb;
import shapely.geometry;
import shapely.ops;
import pyproj;
#from mpl_toolkits.basemap import Basemap;
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection;
import sys;
from descartes import PolygonPatch;
from functools import partial;


def colorclamp(inp):
 inp = inp / 200;
 if inp > 1.0:
  inp = 1.0;
 return inp
 


# Extract first layer of features from shapefile using OGR
ds = ogr.Open(sys.argv[1])
layer = ds.ExecuteSQL("SELECT * from polylayer WHERE CountyID = 49035")
#nlay = ds.GetLayerCount()
#layer = ds.GetLayer(0)

project = pyproj.Proj(init='epsg:26912')

#srs.SetWellKnownGeogCS("EPSG:32612")

# Get extent and calculate buffer size
extent = layer.GetExtent()
xoff = (extent[1]-extent[0])/50
yoff = (extent[3]-extent[2])/50

print(extent)

#exit()

# = Basemap(projection='stere',lat_0=extent[2],lon_0=extent[0],llcrnrlon=extent[0]-1,llcrnrlat=extent[2]-1,urcrnrlon=extent[1]+1,urcrnrlat=extent[3]+1)
#.drawmapboundary();
#.drawstates();
#.readshapefile(sys.argv[1],'blah')
#plt.show()
#exit();

x0,y0=project(extent[0]-xoff, extent[2]-yoff)
x1,y1=project(extent[1]+xoff, extent[3]+yoff)

#aspect = abs(x1-x0)/abs(y1-y0)
#print(aspect)

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(x0,x1)
ax.set_ylim(y0,y1)
#ax.set_xlim(-200000,200000)
#ax.set_ylim(-200000,200000)

paths = []
layer.ResetReading()


# Read all features in layer and store as paths
count = 0;
patches = []
c=[]
for feat in layer:
    geom = feat.geometry()
    item = sh.wkb.loads(geom.ExportToWkb());
    #print(type(item));
    #if type(item) == shapely.geometry.polygon.Polygon:
      #ax.add_patch(patch);
    for i in range(feat.GetFieldCount()):
      print(feat.GetFieldDefnRef(i).GetName())
      print(feat.GetFieldDefnRef(i).GetTypeName())
      #pop=feat.GetFieldAsDouble("POP10")
      #it2 = sh.ops.transform(project, item)
      #it2 = item
      
      #patches.append(PolygonPatch(it2,edgecolor='none',linewidth=0.1));
      #c.append(pop)

    break
    #count+=1;
    #if(count > 1000):
    #  break;

exit()

pc=PatchCollection(patches,cmap=matplotlib.cm.Blues)
pc.set_array(np.array(c))
pc.set_linewidth(0.2)
ax.add_collection(pc )
ax.set_aspect(1.0)
plt.show()
#plt.gcf();
#plt.set_size_inches(10,10);
#plt.set_dpi(200);
#plt.savefig("test.png")
