from osgeo import ogr;
from osgeo import osr;
import numpy as np;
import shapely as sh;
import shapely.wkb;
import shapely.geometry;
import shapely.ops;
import pyproj;
import rtree;
import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib.patches;
import os;
from reportlab.platypus import tables
import types
import sys;
import sqlite3;
import collections;



def getBlockPolys(filename):
	"""
	Obtains the polygons from a census block shapefile
	Returns an Rtree index for the polys plus an array for 
	the BLOCKID10 field
	"""
	
	p=rtree.index.Property(variant=rtree.index.RT_Star)
	idx = rtree.index.Rtree(properties=p)
	
	ds = ogr.Open(filename);	
	#this is probably a hack of sort
	layer=ds.GetLayer(0);
	
	sprIn = layer.GetSpatialRef();
	sprOut = osr.SpatialReference()
	sprOut.ImportFromEPSG(26912)
	trans = osr.CoordinateTransformation(sprIn,sprOut);
	
	#extent=layer.GetExtent();
	for feat in layer:
		geom = feat.geometry();
		geom.Transform(trans);
		item = sh.wkb.loads(geom.ExportToWkb());
		id = feat.GetFieldAsInteger64("BLOCKID10");

		idx.insert(id,item.bounds,obj=item);
		
	return idx;
		

def getPointsUT(filename):
	"""
	Loads the points from an address pointfile
	THIS IS NOT GENERALIZED, so later this will need 
	to be adapted for general datasets
	Returns points + full street address + city		
	"""
	points = [];
	addr = [];
	locale = [];
	
	ds = ogr.Open(filename);
	layer = ds.GetLayer(0);
	
	sprIn = layer.GetSpatialRef();
	sprOut = osr.SpatialReference()
	sprOut.ImportFromEPSG(26912)
	trans = osr.CoordinateTransformation(sprIn,sprOut);
	
	for feat in layer:
		if(feat):
			geom = feat.geometry();
			if(geom):
				geom.Transform(trans);
				points.append(sh.wkb.loads(geom.ExportToWkb()));
				addr.append(feat.GetFieldAsString("FullAdd"));
				locale.append(feat.GetFieldAsString("AddSystem"))
	
	return points,addr,locale;
	
#points (actual phys location), (address name), (city locale), (block ID)
def assignPoints(points,index, addr, locale):
	output = {}
	for i in range(len(points)):
		for j in index.intersection( [points[i].x, points[i].y, points[i].x, points[i].y], objects=True ):
			#print(j.obj)
			if(points[i].within(j.object)):
				key = j.id;
				if key in output:
					output[key] += [[points[i],addr[i],locale[i]]];
				else:
					output[key] = [[points[i],addr[i],locale[i]]]
				
	return output;


def plotIndex(idx,name):
	b = idx.bounds;
	b = [b[0],b[2],b[1],b[3]];
	plt.axis(b);
	leafs = idx.leaves();
	for i in leafs:
		plt.gca().add_patch(matplotlib.patches.Rectangle((i[2][0],i[2][1]),i[2][2]-i[2][0],i[2][3]-i[2][1],fill=False))
	plt.gca().set_aspect(1.0)
	plt.savefig(name, dpi=200)
	


'''
For now this is using hardcoded paths until other things can be figured out (maybe a config file that gets read) 
'''
def loadCensusTables():
	
	tables = {};
	
	tablekeys = ["G001", "H3", "P12", "P14","P20", "P22", "P28", "P30", "P31", "P40", "P41", "P43"];
	
	tablefiles = [ 
	"DEC_10_SF1_G001_with_ann.csv",
	"DEC_10_SF1_H3_with_ann.csv",
	"DEC_10_SF1_P12_with_ann.csv",
	"DEC_10_SF1_P14_with_ann.csv",
	"DEC_10_SF1_P20_with_ann.csv",
	"DEC_10_SF1_P22_with_ann.csv",
	"DEC_10_SF1_P28_with_ann.csv",
	"DEC_10_SF1_P30_with_ann.csv",
	"DEC_10_SF1_P31_with_ann.csv",
	"DEC_10_SF1_P40_with_ann.csv",
	"DEC_10_SF1_P41_with_ann.csv",
	"DEC_10_SF1_P43_with_ann.csv"
	]
	
	tabledirs = [ 
	"/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/slccounty/",
	"/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/externalcounties/"
	]
	
	for i in range(len(tablekeys)):
		temptable0 = pd.read_csv(tabledirs[0]+tablefiles[i], skiprows=[1], index_col=[1])
		temptable1 = pd.read_csv(tabledirs[1]+tablefiles[i], skiprows=[1], index_col=[1])
		tables[tablekeys[i]] = pd.concat([temptable0, temptable1]); 
		
	return tables;


#ugliest code EVER. 
#t is for TABLES
def	assignHouseholds(t):

	maxpop = np.sum(t['P12']['D001']);
	cpop = 0;
	chouse = 0;
	
	agebracket=[20,21,22,25,30,35,40,45,50,55,60,62,65,67,70,75,80,85,100];
	frange = [32,49+1]
	mrange = [8,25+1]
	childages = [0,19]
	fcrange = [24,43+1]
	mcrange = [3,22+1]
	
	#format:
	#age, minage, maxage, g1,g2,gender (m=0,f=1),hh,hholder,group,mobile,employed,addrx,addry
	allindvs = pd.DataFrame(index=np.arange(maxpop),columns=['age','minage','maxage','g1','g2','gender','household','group','mobile','employed','block','addrx','addry']);
	allindvs['household'] = -1
	allindvs['g1'] = allindvs.index;
	allindvs['g2'] = allindvs.index;
	allindvs['group'] = int(False);
	allindvs['mobile'] = int(True);
	
	
	
	for i in range(len(t["G001"].index)):
		index = t['G001'].iloc[i].name
		
		print(index);
		
		pop = t['P12'].at[index,'D001'];
		if(pop < 1): continue;
		
		counter = collections.Counter();
		
		#adult females
		for j in enumerate(range(frange[0],frange[1])):
			id = 'D' + str(j[1]).zfill(3);
			key = (agebracket[j[0]],agebracket[j[0]+1],1, index)
			count = t['P12'].at[index,id]
			counter[key] = count;
			
		#adult males
		for j in enumerate(range(mrange[0],mrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			key = (agebracket[j[0]],agebracket[j[0]+1],0, index)
			count = t['P12'].at[index,id]
			counter[key] = count;
			
		#minor females
		for j in enumerate(range(fcrange[0],fcrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			key = (j[0],j[0]+1,1, index)
			count = t['P14'].at[index,id]
			counter[key] = count;
			
		#minor males
		for j in enumerate(range(mcrange[0],mcrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			key = (j[0],j[0]+1,0, index)
			count = t['P14'].at[index,id]
			counter[key] = count;
		
		allindvs.ix[cpop:pop+cpop-1,['minage','maxage','gender','block']] = list(counter.elements())
		
		#NEVER SHUFFLE HOUSEHOLDS
		households = [];
		
		#ORDER OF ASSEMBLY MATTERS
		#non-family is placed first, followed by fmaily
		#secondly, husband wife is placed last because of uncertainty in gender
		#ratios for householders (male or female can be householder in h/w family(
		

				
		#family w/people under 18
		hcount = t['P20'].at[index,'D009'] #mhc
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=True,female=False,child=True,group=False,minage=15,size=0)]
		hcount = t['P20'].at[index,'D013'] #fhc
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=False,female=True,child=True,group=False,minage=15,size=0)]
		
		#family w/o people under 18
		hcount = t['P20'].at[index,'D030'] #mh
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=True,female=False,child=False,group=False,size=0,minage=15)]
		hcount = t['P20'].at[index,'D031'] #fh
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=False,female=True,child=False,group=False,size=0,minage=15)]
			
		#husband wife
		hcount = t['P20'].at[index,'D004'] #hwc
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=True,female=True,child=True,group=False,minage=15,size=0)]
		hcount = t['P20'].at[index,'D028'] #hw
		for n in range(hcount):
			households += [types.SimpleNamespace(family=True,male=True,female=True,child=False,group=False,size=0,minage=15)]
		
		#non family w/children	
		hcount = t['P20'].at[index,'D018'] #nfmc
		for n in range(hcount):
			households += [types.SimpleNamespace(family=False,male=True,female=False,child=True,group=False,size=0,minage=15)]
		hcount = t['P20'].at[index,'D022'] #nffc
		for n in range(hcount):
			households += [types.SimpleNamespace(family=False,male=False,female=True,child=True,group=False,size=0,minage=15)]
		#non family no children
		hcount = t['P20'].at[index,'D033'] #nfm
		for n in range(hcount):
			households += [types.SimpleNamespace(family=False,male=True,female=False,child=False,group=False,size=0,minage=15)]
		hcount = t['P20'].at[index,'D034'] #nff
		for n in range(hcount):
			households += [types.SimpleNamespace(family=False,male=False,female=True,child=False,group=False,size=0,minage=15)]
		
		#deal with group housing separately
		#group housing is separate from other households and must be treated differently
		if t['P43'].at[index,'D001'] > 0:
			households += [types.SimpleNamespace(family=False,male=False,female=False,child=False,group=True,size=0,minage=0)]
	
		
		
	
	
		cpop += pop;
		chouse += len(households);
		
		if(cpop > 10000):
			break;
	
	print(allindvs[0:1000]);


	return allindvs;
		
		
		
		#indiv
	
#FIXME: I probably need to assign grou phousing differently
#(it doesn't make much sense to have 300 inmates in a standard house
#on the other hand, do prisons have their own census block?
def assignAddresses(indvs, addresses):
	
	for i in range(len(indvs)):
# 		print(indvs[i])
		if(len(indvs[i]) > 0):
			index = indvs[i][0].block
			if(index in addresses):
	# 			print(index)
				blockaddr = [ind for ind, x in enumerate(addresses[index])];
				maxblock = len(blockaddr);
				#blockindvs = [ind for ind, x in enumerate(indvs) if x.block == i];
				if( len(blockaddr) > 0 and len(indvs[i]) > 0):
					housemap = [x.household for x in indvs[i]]				
					np.random.shuffle(blockaddr);
					for j in range(len(indvs[i])):
						indvs[i][j].address = addresses[index][blockaddr[housemap[j] % maxblock]];
			
			
# 			print(blockaddr)
# 			print(housemap)
# 			print( [indvs[x].address for x in blockindvs] )
# 		print("break")
	return indvs;
	
def plotIndvs(indvs):
	
	
	#transform to a nice useable format
	vals = [];
	for i in range(len(indvs)):
		for j in range(len(indvs[i])):
			if hasattr(indvs[i][j],'address'):
				group = '#FA3532'; #red
				if(indvs[i][j].age > 64):
					group = '#FFB300' #orange
				if(indvs[i][j].age < 18):
					group = '#1EA3E6' #blue
				pt = indvs[i][j].address[0];
				vals += [[(pt.x + (np.random.randn()*10.0)),(pt.y + (np.random.randn()*10.0)),group]]
	
	v = np.array(vals)
	v = v.T
	#plot nicely
	
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect = 'equal');
	plt.scatter(v[0],v[1],c=v[2],s=1,lw=0.0)
	#plt.show()
	F = plt.gcf();
	F.set_size_inches(200,200)
	F.set_dpi(100);
	F.savefig("indvmap.png");
	
	

def main():
	
	print("Loading census tables...")
	tables = loadCensusTables()
	print("Assigning household structure...");
	indvs = assignHouseholds(tables)
	
	con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/indvs.sq3");
	indvs.to_sql('indvs',con);
	con.close();
	
	exit();
	os.environ['GDAL_DATA']='/uufs/chpc.utah.edu/common/home/u0403692/anaconda/share/gdal/'
	print("Loading address points...")
	pts,addr,loc = getPointsUT("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/AddressPoints/AddressPoints.shp")
	#pts,addr,loc = getPointsUT("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/small/small_points.shp")
	print("Loading census blocks...")
	idx = getBlockPolys("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/tabblock2010_49_pophu/tabblock2010_49_pophu.shp")
	#idx = getBlockPolys("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/small/small_shapes.shp")
	#	print(pts,addr,loc);
	print("Assigning points...")
	addresses = assignPoints(pts,idx,addr,loc);
	#exit()
	print("Assigning home addresses to indviduals...");
	indvs2 = assignAddresses(indvs,addresses);
	print("Writing to file...")
	out = pd.DataFrame(indvs2)
	out.to_pickle("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/assigned.pickle")
	print("Plotting locations...")
	plotIndvs(indvs2)
		
	
if __name__ == "__main__":
	main()
	