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



def getBlockPolys(filename):
	"""
	Obtains the polygons from a census block shapefile
	Returns an Rtree index for the polys plus an array for 
	the BLOCKID10 field
	"""
	
	p=rtree.index.Property(variant=rtree.index.RT_Star)
	idx = rtree.index.Rtree(properties=p)
	
	alts = {}
	
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
		alts[id] = item.centroid
		
	return idx,alts;
		

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

	#indices = [];
	allindvs = []
	cc = 0;
	
	agebracket=[20,21,22,25,30,35,40,45,50,55,60,62,65,67,70,75,80,85,100];
	frange = [32,49+1]
	mrange = [8,25+1]
	childages = [0,19]
	fcrange = [24,43+1]
	mcrange = [3,22+1]
	
	cpop = 0;
	hpop = 0;
	
	for i in range(len(t["G001"].index)):
		
		index = t['G001'].iloc[i].name
		
# 		if(index != 490351002001020):
# 			continue;
# 		if(index != 490519405004038):
# 			continue;
		
		#indices += [index]
# 		print(index)
		#assign individuals
		pop = t['P12'].at[index,'D001'];

		#print(pop)
		
		

		
		indvs = [];
		
		#assign adult females
		for j in enumerate(range(frange[0],frange[1])):
			id = 'D' + str(j[1]).zfill(3);
			count = t['P12'].at[index,id]
			for k in range(count):
				x = types.SimpleNamespace()
				#x.age = np.random.randint(agebracket[j[0]],agebracket[j[0]+1])
				x.age = set(range(agebracket[j[0]],agebracket[j[0]+1]))
				#x.bracket = j[0]
				x.block = index
				x.g1 = len(indvs)
				x.g2 = len(indvs)
				x.gen = 'f'
				x.household = -1;
				x.householder = 0;
				x.group = False;
				x.mobile = True;
				x.spouse = -1;
				indvs+=[x]
		
		#assign adult males		
		for j in enumerate(range(mrange[0],mrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			count = t['P12'].at[index,id]
			for k in range(count):
				x = types.SimpleNamespace()
				#x.age = np.random.randint(agebracket[j[0]],agebracket[j[0]+1])
				x.age = set(range(agebracket[j[0]],agebracket[j[0]+1]))
				#x.bracket = j[0]
				x.block = index
				x.g1 = len(indvs)
				x.g2 = len(indvs)
				x.gen = 'm'
				x.household = -1;
				x.householder = 0;
				x.group = False;
				x.mobile = True;
				x.spouse = -1;
				indvs+=[x]
						
		#assign female children
		for j in enumerate(range(fcrange[0],fcrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			count = t['P14'].at[index,id]
			for k in range(count):
				x = types.SimpleNamespace()
				x.age = {j[0]}
				x.block = index
				x.g1 = len(indvs)
				x.g2 = len(indvs)
				x.gen = 'f'
				x.household = -1;
				x.householder = 0;
				x.group = False;
				x.mobile = True;
				x.spouse = -1;
				indvs+=[x]
				
		#assign male children
		for j in enumerate(range(mcrange[0],mcrange[1])):
			id = 'D' + str(j[1]).zfill(3);
			count = t['P14'].at[index,id]
			for k in range(count):
				x = types.SimpleNamespace()
				x.age = {j[0]}
				x.block = index
				x.g1 = len(indvs)
				x.g2 = len(indvs)
				x.gen = 'm'
				x.household = -1;
				x.householder = 0;
				x.group = False;
				x.mobile = True;
				x.spouse = -1;
				indvs+=[x]
		
		#pick householders
		#households = np.zeros(t['H3'].at[index,'D002']);
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
		

	#assign children to households
	#assignment is by round robin because we must guarantee 
	# that every household that requires children also receives 
	#children
	#because we know the allocations between different key family types
	#the average family size will be accurate to within the limits of the
	#block demographics
	#this also guarantees that for every household with children
	#the household profile for exposure is taken into account
	
	
		chbracket = [ 0,3,5,6,12,18 ] 
# 		print(households)
		hind = [ ind for ind, x in enumerate(households) if x.family and x.child ] 
		#print(hind)
		#print( [ households[ind] for ind in hind] )
		np.random.shuffle(hind)
		#assign grand children, if any	
		for n in enumerate(range(2,6+1)):
# 			print(n)
			id = 'D' + str(n[1]).zfill(3);
			count = t['P41'].at[index,id]
# 			print("g",count)
			if count > 0:
				minmax = range(chbracket[n[0]],chbracket[n[0]+1])
				childind = [  ind for ind, x in enumerate(indvs) if x.age.intersection(minmax) and x.household < 0 ]
				np.random.shuffle(childind)
				newhh = np.random.randint(0, len(hind))
				for nn in range(count):
					indvs[childind[nn]].household = hind[newhh];
					households[hind[newhh]].minage = max(30 + list(indvs[childind[nn]].age)[0], households[hind[newhh]].minage)
					households[hind[newhh]].size += 1
					newhh = (newhh + 1) % len(hind)
		
		#assign family children		
# 		print(indvs)
		for d in [(3,True,True),(10,True,False),(16,False,True)]:
			hind = [ ind for ind, x in enumerate(households) if x.family and x.child and x.male == d[1] and x.female == d[2] ]
			np.random.shuffle(hind)
# 			print(d)
			for n in enumerate(range(d[0],d[0]+5)):
# 				print(n)
				id = 'D' + str(n[1]).zfill(3);
				count = t['P40'].at[index,id]
# 				print("f",count)
				if count > 0:
					minmax = range(chbracket[n[0]],chbracket[n[0]+1])
# 					print(minmax)
					childind = [  ind for ind, x in enumerate(indvs) if x.age.intersection(minmax) and x.household < 0 ]
# 					print(childind)
# 					print([indvs[c] for c in childind])
					np.random.shuffle(childind)
					newhh = np.random.randint(0, len(hind))
					for nn in range(count):
						indvs[childind[nn]].household = hind[newhh];
						households[hind[newhh]].minage = max(15 + list(indvs[childind[nn]].age)[0], households[hind[newhh]].minage)
						households[hind[newhh]].size += 1
						newhh = (newhh + 1) % len(hind)
		
		#the weird edge case of 490351008003005
		#in this situation there aren't any households that have children
		#but there are children in group housing, so mis-assignment occurs.
		
		#!!!! Juvenile facilities include juvenile prison AND other group housing
		#for the purposes of this simulation juvenile facilities will count as 
		#restricted movement facilities even if they may allow movement
		
		if t['P43'].at[index,'D001'] > 0:
			
			#this data combines both genders. because the distribution of children to family households
			#is not divided by gender random assignment can results in situations where no enough 
			#unassigned children remain in a particular age bracket. 
			#since gender is only secondary (only matters for assigning householders) this is a non-issue. 
			
			#get the number of group housing children
			icount = t['P43'].at[index,'D004'] + t['P43'].at[index,'D035'] #institutionalized
			pcount = t['P43'].at[index,'D005'] + t['P43'].at[index,'D036'] #+ t['P43'].at[index,'D006'] #prison/incarcerated
			ncount = t['P43'].at[index,'D009'] + t['P43'].at[index,'D040'] #non-institutionalized
			mcount = t['P43'].at[index,'D011'] + t['P43'].at[index,'D042'] #military
			
			#99% of incarcerated children are ages 13 and up
			#extremely unlikely (500/300,000,000) to find an instance where 12 and under incarecerateion occurs
			#HOWEVER, because minor assignment has a bracket in the 12-17 range, there may be instances where 
			#not enough individuals are avaiable after household assignment because the brackets do not cleanly overlap
			minmax = range(12,18)
			hind = [ ind for ind, x in enumerate(households) if x.group ]; #kind of silly, always will have one element
			
			childind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
# 			print([indvs[x] for x in childind]);
# 			print([x for x in indvs if x.household < 0]) 
			np.random.shuffle(childind);
			if(pcount > 0):
				for nn in range(pcount):
					indvs[childind[nn]].household = hind[0];
					indvs[childind[nn]].mobile = False;
					indvs[childind[nn]].group = True;
					households[hind[0]].size += 1
					
			#other institutionalized children
			minmax = range(0,18)
			
			childind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
			np.random.shuffle(childind);
			count = (icount - pcount) + mcount
			if(count > 0):
				for nn in range(count):
					indvs[childind[nn]].household = hind[0];
					indvs[childind[nn]].mobile = False;
					indvs[childind[nn]].group = True;
					households[hind[0]].size += 1
					
			#non-institutionalized children
			childind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
			np.random.shuffle(childind);
			count = ncount - mcount
			if(count > 0):
				for nn in range(count):
					indvs[childind[nn]].household = hind[0];
					indvs[childind[nn]].mobile = True;
					indvs[childind[nn]].group = True;
					households[hind[0]].size += 1

		#assign remaining children
		#this SHOULD happen after householder assignment because 
		#kids 15 and up can be classified as householders.
		#but, because of some tangle with the guardianship assignment it still needs to be first
		minmax = range(0,18) # age < 18
		hind = [ ind for ind, x in enumerate(households) if x.child and x.size < 1 ] + [ ind for ind, x in enumerate(households) if x.child and x.size > 0 ]
		childind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
		
		# 		print(childind)
		if len(childind) > 0:
			np.random.shuffle(childind)
			newhh = 0 #because we need to make sure all child required households get one
			for nn in range(len(childind)):
				indvs[childind[nn]].household = hind[newhh];
				households[hind[newhh]].minage = max(15 + list(indvs[childind[nn]].age)[0], households[hind[newhh]].minage)
				households[hind[newhh]].size += 1
				newhh = (newhh + 1) % len(hind)
		

		
		#print([x for x in households if x.size > 0])
		#print([households])
		#print([x for x in indvs if x.age > 18])

		#  it would be nice in a future revision to also incorporate householder ages, but this is complicated
		#assign householder adults
# 		print(len(households))
		for h in range(len(households)):
			
			#ignore group housing
			if(households[h].group):
				continue;
			#gender select
			g = 'fm'
			if households[h].male and (not households[h].female):
				g = 'm'
			elif (not households[h].male) and (households[h].female):
				g = 'f'
			
			#the absolute minimum age of a householder is 15 per the
			#census bureau
			minage = 15; maxage = 100;
			#if there are children in the household we need to adjust the minimum age
			if(households[h].child and households[h].family):
				minage = households[h].minage;
				maxage = minage + 15; 
			
			#print('?',minage, maxage, [x.age for x in indvs if x.household < 0],[(x.age,x.gen) for x in indvs if x.household == h]);
# 			print(minage, maxage)
			for cutoff in [0,10,15,25,30,35]:
				minmax = range(minage,maxage+cutoff)
				hhcand = [ind for ind,x in enumerate(indvs) if x.age.intersection(minmax) and (x.gen in g) and x.household < 0];
				if(len(hhcand) > 0): break;
			
			#if we get to this point and the hhcandlist is still empty, it implies that 
			#there is a 15-17 yo householder who has already been assigned to the household
			#!!! HOW DO WE KNOW THAT THE HOUSEHOLDER HAS ALREADY BEEN ASSIGNED?!
			if(len(hhcand) < 1):
				hhcand = [ind for ind,x in enumerate(indvs) if x.age.intersection(range(15,18)) and (x.gen in g)];
						
			s = np.random.randint(0,len(hhcand));
			select = hhcand[s];
			
			indvs[select].householder = 1;
			indvs[select].household = h;
			households[h].size += 1;
			households[h].hh = select;
			

# 		print([(x.age,x.gen,x.household,x.householder) for x in indvs])

		#assign "spouse"
		# gender disparity in spouses skews to older husbands
		spupper=[1,5,3,7]
		splower=[-1,-5,-10,-15]
		for h in range(len(households)):
			if households[h].male and households[h].female:
# 				print(h);
# 				print([x for x in indvs if x.household < 0])
# 				print([x for x in indvs if x.household == h])
				sg = 'm';
				#if female spouse, swap gender and boundary limits
				if(indvs[households[h].hh].gen == sg): 
					sg = 'f';
					temp = spupper
					spupper = [-x for x in splower]
					splower = [-x for x in temp]
				for k in range(0,4):
					minmax = range(min(indvs[households[h].hh].age) + splower[k],max(indvs[households[h].hh].age) + spupper[k])
					hhcand = [ind for ind,x in enumerate(indvs) if x.age.intersection(minmax) and (x.gen in sg) and x.household < 0];
					if(len(hhcand) > 0): break;
				if(len(hhcand) < 1): hhcand = [ind for ind,x in enumerate(indvs) if x.age.intersection(range(18,100)) and (x.gen in sg) and x.household < 0];
				#edge case: if no available spouses remain in the 18,100 bracket we find a reassign
				#one from a different family unit
				if(len(hhcand) < 1): hhcand = [ind for ind,x in enumerate(indvs) if x.age.intersection(range(15,18)) and (x.gen in sg)];

				s = np.random.randint(0,len(hhcand));
				spselect = hhcand[s]#
				indvs[spselect].household = h;
				households[h].size += 1;
				indvs[spselect].spouse = households[h].hh
				indvs[households[h].hh].spouse = spselect;

		

		
		
		#assign guardians to children
		for h in range(len(households)):
			if households[h].child:
				guard1 = households[h].hh;
				if households[h].male and households[h].female:
					candlist = [ind for ind,x in enumerate(indvs) if x.household == h and x.householder < 1 and x.age.intersection(range(18,100))]
					if(len(candlist) < 1):
						[ind for ind,x in enumerate(indvs) if x.household == h and x.householder < 1 and x.age.intersection(range(15,18))]
					if(len(candlist) < 1):
						guard2=guard1;
					else:
						guard2 = candlist[0]
				else:
					guard2 = guard1 
				childlist = [ind for ind,x in enumerate(indvs) if x.household == h and x.age.intersection(range(0,18))]
				for c in childlist:
					indvs[c].g1 = guard1;
					indvs[c].g2 = guard2;
		
		
		#assign nonmobile group populations
		if t['P43'].at[index,'D001'] > 0:
			
			for i in range(0,2):
				
				#18-64
				if i == 0:
					minmax = range(18,65);
					military = t['P43'].at[index,'D021'] + t['P43'].at[index,'D052']
					nonmobcount = t['P43'].at[index,'D014'] + t['P43'].at[index,'D045'] + military #institutionalized + military
					mobcount = t['P43'].at[index,'D019'] + t['P43'].at[index,'D050'] - military #noninstitutionalized - military
				
				if i == 1:
				#65-100
					minmax = range(65,100);
					military = t['P43'].at[index,'D031'] + t['P43'].at[index,'D062']
					nonmobcount = t['P43'].at[index,'D024'] + t['P43'].at[index,'D055'] + military #institutionalized + military
					mobcount = t['P43'].at[index,'D029'] + t['P43'].at[index,'D060'] - military #noninstitutionalized - military
				
				hind = [ ind for ind, x in enumerate(households) if x.group ];
				#non-mobile
				if(nonmobcount > 0):
					adultind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
					if(len(adultind) < nonmobcount):
							adultind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(range(18,100)) ]
					np.random.shuffle(adultind);
					for nn in range(nonmobcount):
						indvs[adultind[nn]].household = hind[0];
						indvs[adultind[nn]].mobile = False;
						indvs[adultind[nn]].group = True;
						households[hind[0]].size += 1
						
				#mobile
				if(mobcount > 0):
					adultind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(minmax) ]
					if(len(adultind) < mobcount):
							adultind = [  ind for ind, x in enumerate(indvs) if x.household < 0 and x.age.intersection(range(18,100)) ]
					np.random.shuffle(adultind);
					for nn in range(mobcount):
						indvs[adultind[nn]].household = hind[0];
						indvs[adultind[nn]].mobile = True;
						indvs[adultind[nn]].group = True;
						households[hind[0]].size += 1
		
		
		#assign remaining adults
		
		
#this code was problematic for unknown reasons
# 		#if a family house, make sure the number of residents is at least 2 
# 		rem = [ind for ind,x in enumerate(indvs) if x.household < 0];
# 		np.random.shuffle(rem);
# 		famhouse = [ind for ind,x in enumerate(households) if x.size < 2 and x.family]
# # 		print(rem, famhouse)
# # 		for f in range(len(famhouse)):
# # 			indvs[rem[f]].household = famhouse[f];
# # 			households[famhouse[f]].size += 1;
		
		rem = [ind for ind,x in enumerate(indvs) if x.household < 0];
		for r in rem:
			select = np.random.randint(0,len(households))
			indvs[r].household = select;
			households[select].size += 1;
			
			
		

		#select ages and update IDs
# 		addin = len(allindvs)
		for i in range(len(indvs)):
			ages = list(indvs[i].age)
			if(len(ages) == 1):
				indvs[i].age = ages[0]
			else:
				select = np.random.randint(0,len(ages));
				indvs[i].age = ages[select];
			
			indvs[i].id = i + cpop;
			indvs[i].household += hpop;
			indvs[i].g1 += cpop;
			indvs[i].g2 += cpop;
			if(indvs[i].spouse > -1): indvs[i].spouse += cpop;
				
# 		#print([ (x.age,x.household,x.householder) for x in indvs ])
# 		if(index == 490351008003005):
# 			print(indvs)
# 		print(households)
		
		cpop += pop;
		hpop += len(households);
		
		allindvs += [indvs];
# 		cc+=1;
# 		if(cc > 100):
# 			break;
# 		else:
# 			print("breark")
		
	return allindvs;
		
		
		
		#indiv
	
#FIXME: I probably need to assign grou phousing differently
#(it doesn't make much sense to have 300 inmates in a standard house
#on the other hand, do prisons have their own census block?
def assignAddresses(indvs, addresses,alts):
	
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
			else: 
				print("BAD:",index, end=' ')
				badaddr =[alts[index], 'BAD','BAD']
				for j in range(len(indvs[i])):
					indvs[i][j].address = badaddr;
				
				
				
			
			
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


#converts the STUPID NAMESPACE FORMAT 
#into a useable DataFrame
def saveData(indvs):
	
	#get size;

	print('gen data frame')
	allindvs = pd.DataFrame([y for x in indvs for y in x]);
	#print(allindvs);
	allindvs.columns = ['raw']


	print('transform dataframe')
	g = lambda x: 1 if(x=='f') else 0;
	tf = lambda x: 1 if(x==True) else 0;

	expr = [ 
		('age', lambda b:b.age ),
		('g1', lambda b:b.g1 ),
		('g2', lambda b:b.g2 ),
		('gender', lambda b:g(b.gen) ),
		('household', lambda b:b.household ),
		('householder', lambda b:b.householder ),
		('group', lambda b:tf(b.group) ),
		('mobile', lambda b:tf(b.mobile) ),
		('block', lambda b:b.block ),
		('addrx', lambda b:b.address[0].x ),
		('addry', lambda b:b.address[0].y ),
		('addrn', lambda b:b.address[1] ),
		('city', lambda b:b.address[2] ),
		('id', lambda b:b.id ),
		('spouse',lambda b:b.spouse ),
	]
	#cpop = 0;
	for i in expr:
		allindvs[i[0]] = allindvs['raw'].apply(i[1]);
	
	allindvs = allindvs.drop('raw',axis=1);
	allindvs.set_index('id')
			
	return allindvs;

def convaddr(addrs):
	
	df = pd.DataFrame([tuple((key,)+tuple(y)) for key,item in addrs.items() for y in addrs[key] ])
	
	df.columns = ['block','rawpoint','addrn','city']
	
	df['addrx'] = df['rawpoint'].apply(lambda b:b.x)
	df['addry'] = df['rawpoint'].apply(lambda b:b.y)
	df = df.drop('rawpoint',axis=1)
	
	return df;
	  

def main():
	
	print("Loading census tables...")
	tables = loadCensusTables()
	print("Assigning household structure...");
	indvs = assignHouseholds(tables)
	
	#saveData(indvs);
	
	os.environ['GDAL_DATA']='/uufs/chpc.utah.edu/common/home/u0403692/anaconda/share/gdal/'
	print("Loading address points...")
	pts,addr,loc = getPointsUT("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/AddressPoints/AddressPoints.shp")
	#pts,addr,loc = getPointsUT("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/small/small_points.shp")
	print("Loading census blocks...")
	idx,alts = getBlockPolys("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/tabblock2010_49_pophu/tabblock2010_49_pophu.shp")
	#idx,alts = getBlockPolys("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/small/small_shapes.shp")
	#	print(pts,addr,loc);
	print("Assigning points...")
	addresses = assignPoints(pts,idx,addr,loc);
	
	outaddr = convaddr(addresses);
	con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/blockaddr2.sq3");
	outaddr.to_sql('blockaddr',con);
	con.close();

	print("Assigning home addresses to indviduals...");
	indvs = assignAddresses(indvs,addresses,alts);
	print("Prepping for write...")
	indvs = saveData(indvs);
	print("Writing to file...")
	con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/indvs2.sq3");
	indvs.to_sql('indvs',con);
	con.close();
	#out.to_pickle("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/assigned.pickle")
	#print("Plotting locations...")
	#plotIndvs(indvs2)
		
	
if __name__ == "__main__":
	main()
	