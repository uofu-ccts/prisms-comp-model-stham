import rtree;
import pandas as pd;
import numpy as np;
from shapely import wkt;
# import shapely.geometry;
# import shapely.ops;
import pyproj;
import sqlite3;


outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');

def reversetransraw(locx,locy):
	locx,locy=pyproj.transform(outproj,inproj,locx,locy);
	return locx,locy


def buildloctree(frame):
	p=rtree.index.Property(variant=rtree.index.RT_Star)
	idx = rtree.index.Rtree(properties=p)
	for ind,i in frame.iterrows():
		idx.insert(ind,(i.x,i.y,i.x,i.y));
	return idx

def manglewkb(x):
	d = wkt.loads(x.WKT_GEOMETRY);
	lx,ly = reversetransraw(d.x,d.y);
	return lx,ly

# p=rtree.index.Property(variant=rtree.index.RT_Star)
# idx = rtree.index.Rtree(properties=p)

def main():
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

	s = ""
	taglist = ['office','shop','amenity','tourism','leisure','sport']
	for ind,i in enumerate(taglist):
		s += "other_tags like \'%\"" + i + "\"=>%\'"
		if(ind < (len(taglist)-1)):
			s += " or "

	query = "select WKT_GEOMETRY from utahosmlatest where " + s + ";"

	print(query)


	print("read and build frame")
	con = sqlite3.connect(datapath + "utahosmlatest.sqlite");
	frame = pd.read_sql(query, con);
	con.close();

	frame['x'],frame['y'] = zip(*frame.apply(manglewkb,axis=1));

	frame.to_csv(datapath + "randlocs.csv");

	# print("building rtree")
	# idx = buildloctree(frame);

if __name__ == "__main__":
	main();



