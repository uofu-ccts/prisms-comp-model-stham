import pandas as pd;
import numpy as np;
import sqlite3;
import collections;
from scipy.stats import exponweib;
from scipy.stats import norm;
import datetime
import time;
import multiprocessing as mp;
import matplotlib.pyplot as plt;
import pyproj;
import h5py;
import mkl;
import sys;
import pycurl;
from io import BytesIO;
import json;
import resource;
import rtree;

#import osrm;

# sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/mathmodel/python");
# import activitydecision as ad;


locprobtab = np.array([ 0.02666667,  0.02639456,  0.02612245,  0.02585034,  0.02557823, 0.02530612,0.02503401,  0.0247619 , 0.0244898 ,  0.02421769, 0.02394558,  0.02367347, 0.02340136,  0.02312925,  0.02285714, 0.02258503,  0.02231293,  0.02204082,  0.02176871, 0.0214966 , 0.02122449,0.02095238,  0.02068027,  0.02040816,  0.02013605, 0.01986395, 0.01959184,  0.01931973,  0.01904762,  0.01877551, 0.0185034 ,  0.01823129,  0.01795918, 0.01768707,  0.01741497, 0.01714286,0.01687075,  0.01659864,  0.01632653,  0.01605442, 0.01578231,  0.0155102 ,  0.0152381 , 0.01496599,  0.01469388, 0.01442177,  0.01414966,  0.01387755,  0.01360544,  0.01333333])


def buildloctree(frame):
	p=rtree.index.Property(variant=rtree.index.RT_Star)
	idx = rtree.index.Rtree(properties=p)
	for ind,i in frame.iterrows():
		idx.insert(ind,(i.x,i.y,i.x,i.y));
	return idx


#wac = pd.read_csv(datapath + "lehd/ut_wac_S000_JT00_2014.csv")

#START CLASSIFICATION FUNCTIONS

def oldnonmobile(x):
	#single activity, 
	return [ (1440,0, x['addrx'],x['addry']) ];

def nonmobile(x):

	out = pd.DataFrame([ (0,1439,1439,0, x['addrx'],x['addry'],x['id']) ])

	out.columns = ['start','length','end','actind','locx','locy','agentnum']

	return out;


def adult(frame, tpgroup, weibull, day):
# 	empexclude = False;
# 	if frame['empcode'] > -1:
# 		empexclude = True;
	#revisit later
	
	actlist = []
	t = np.floor(exponweib.rvs(1.0,1.8,scale=90)) + 120 
	act = 10101
	locx = frame['addrx'];
	locy = frame['addry']
	actlist += [(t,act,locx,locy)]
	
	ct = t;
	while ct < 1440.0:
		

		
		hour = np.floor(ct/60.0)
		g = tpgroup.get_group((day,hour));
		
		#act = np.random.choice(g['dest'],p=(g['count']/np.sum(g['count'])) );
		act = np.random.choice(g['dest'],p=g['prob'] );
		
		#handle movement
		if act <= 189999 and act >= 180000:
			#if we're travelling, we know we start at the current location
			#so locx and locy remain the same
			#but then we have to pick the next item
			wb = weibull.ix[act]
			t = np.floor(exponweib.rvs(1.0,wb['c'],scale=wb['scale'])) + 1
			ct += t;
			actlist += [(t,act,locx,locy)]
			
			#pick based on the type of travel and consider where we currently are
			
			#determine maximum range based on travel time
			#we assume an average speed of 60 kph using taxicab movement
			#60 kph is 1000 m/min
			dist = np.floor(800.0 * t)

			
			#if we are working and we are NOT at work, go to work
			#if we are at work, go home
			if frame['empcode'] > -1 and act >= 180501 and act <= 180599:
				
				if locx != frame['empx'] and locy != frame['empy']:
					locx = frame['empx']; locy = frame['empy']
				else:
					locx = frame['addrx']; locy = frame['addry']
			 	
			#if we are not travelling related to work, then 
			#we we are travelling somewhere else
			#but, we might not travel directly home		
			else:
				if locx == frame['addrx'] and locy == frame['addry']:
					locx = np.random.randint(-dist, dist) + locx;
					locy = np.random.randint(-dist, dist) + locy;
				#after 4 PM 50% probability of going home
				elif t > 720 and (np.random.rand() > 0.5):
					locx = frame['addrx']; locy = frame['addry']
				#after 8 PM 75% probability of going home
				elif t > 960 and (np.random.rand() > 0.25):
					locx = frame['addrx']; locy = frame['addry']
				else:
					locx = np.random.randint(-dist, dist) + frame['addrx'];
					locy = np.random.randint(-dist, dist) + frame['addry'];
			
			g = tpgroup.get_group((day,hour));
			#act = np.random.choice(g['dest'],p=(g['count']/np.sum(g['count'])));
			act = np.random.choice(g['dest'],p=g['prob'] );
		
		#teleport home if its after midnight
		if t > 1200 and locx != frame['addrx'] and locy != frame['addry']:
			locx = frame['addrx']; locy = frame['addry']
			
		wb = weibull.ix[act]
		t = np.floor(exponweib.rvs(1.0,wb['c'],scale=wb['scale'])) + 1
		ct += t;
		actlist += [(t,act,locx,locy)]
			
	return actlist;		
	#construct

def indchild(frame, tpgroup, weibull,day):
	#we assume that school happens around 8:30 on a weekday for 6.5 hours
	school = False;
	if day > 1 and day < 7:
		school = True;
	if school:
		if frame['age'] > 12: grid = 4000.0
		elif frame['age'] > 10: grid = 2000.0
		else: grid = 1000.0;
		
		
	actlist = []
	t = np.floor(exponweib.rvs(1.0,1.8,scale=90)) + 120 
	act = 10101
	locx = frame['addrx'];
	locy = frame['addry']
	actlist += [(t,act,locx,locy)]
	
	ct = t;
	while ct < 1440.0:
		

		
		#force school time
		if school and t > (4.5 * 60.0 - 15) and t < (11.0*60.0 + 15) and act != 60101:
			act = 180601
			t = np.floor(exponweib.rvs(1.0,1.5,scale=15)) + 1
			ct += t;
			locx = frame['addrx']
			locy = frame['addry'] 
			actlist += [(t,act,locx,locy)]
			t = (11.0 * 60.0) - ct;
			ct += t
			act = 60101
			locx = np.round(frame['addrx'] / grid,0)*grid;
			locy = np.round(frame['addry'] / grid,0)*grid;
			actlist += [(t,act,locx,locy)]
		else:
			hour = np.floor(ct/60.0)
			g = tpgroup.get_group((day,hour));
			#act = np.random.choice(g['dest'],p=(g['count']/np.sum(g['count'])));
			act = np.random.choice(g['dest'],p=g['prob'] );
			
			
			#handle movement
			if act <= 189999 and act >= 180000:
				#if we're travelling, we know we start at the current location
				#so locx and locy remain the same
				#but then we have to pick the next item
				wb = weibull.ix[act]
				t = np.floor(exponweib.rvs(1.0,wb['c'],scale=wb['scale'])) + 1
				ct += t;
				actlist += [(t,act,locx,locy)]
				
				#pick based on the type of travel and consider where we currently are
				
				#determine maximum range based on travel time
				#we assume an average speed of 60 kph using taxicab movement
				#30 mph is 800 m/min
				dist = np.floor(800.0 * t)

				
				#if we are working and we are NOT at work, go to work
				#if we are at work, go home
				if frame['empcode'] > -1 and act >= 180501 and act <= 180599:
					
					if locx != frame['empx'] and locy != frame['empy']:
						locx = frame['empx']; locy = frame['empy']
					else:
						locx = frame['addrx']; locy = frame['addry']
				 	
				#if we are not travelling related to work, then 
				#we we are travelling somewhere else
				#but, we might not travel directly home		
				else:
					if locx == frame['addrx'] and locy == frame['addry']:
						locx = np.random.randint(-dist, dist) + locx;
						locy = np.random.randint(-dist, dist) + locy;
					#after 4 PM 50% probability of going home
					elif t > 720 and (np.random.rand() > 0.5):
						locx = frame['addrx']; locy = frame['addry']
					#after 8 PM 75% probability of going home
					elif t > 960 and (np.random.rand() > 0.25):
						locx = frame['addrx']; locy = frame['addry']
					else:
						locx = np.random.randint(-dist, dist) + frame['addrx'];
						locy = np.random.randint(-dist, dist) + frame['addry'];
				
				g = tpgroup.get_group((day,hour));
				
				#act = np.random.choice(g['dest'],p=(g['count']/np.sum(g['count'])));
				act = np.random.choice(g['dest'],p=g['prob'] );

			#teleport home if its after midnight
			if t > 1200 and locx != frame['addrx'] and locy != frame['addry']:
				locx = frame['addrx']; locy = frame['addry']
				
			wb = weibull.ix[act]
			t = np.floor(exponweib.rvs(1.0,wb['c'],scale=wb['scale'])) + 1
			ct += t;
			actlist += [(t,act,locx,locy)]	
			
	return actlist;
			


#how to defer?
#WE PRETEND THAT THE CHILD IS AN ADULT
def depchild(frame,tables, day):
	return adult(frame,tables, day);


def rollweibull(act,tab):
	wb = tab.ix[act]
	return np.floor(exponweib.rvs(1.0,wb['c'],scale=wb['scale'])) + 1

def pickwork(x):
	#shiftclasses:
	#-1 -  no shift
	#0 - normal: 9 am to 5 pm or thereabouts
	#1 - evening: 2 pm to midnight
	#2 - night: 9 pm to 8 am
	#3 - rotating: could be day, evening, night
	#4 - split: two daytime shifts, but this is incovenient so irregular
	#5 - irregular: totally random
	#6 - other: ill defined, so roll into irregular
	if(x > 3): return np.random.randint(0,1440);
	if(x == 3): x = np.random.randint(0,3);
	a = {0:300, 1:600, 2:1020}
	return a[x];

def pickschool(x):
	if(x < 4): return 270;
	return 210 + (60 * np.random.randint(0,7))




def locApply(x,frame,idx):

	# print(x,fr)
	# print(frame.addrx, frame.addry,frame.empx,frame.empy,frame.emphours,x.locp)
	# if(frame.empx == 0.0 and frame.empy == 0.0):
	# 	frame.empx,frame.empy = frame.addrx,frame.addry

	# if(x.locp in [12,13,14,15,16,17,18,19,20,21,99]):
	# 	_locp = x.prevloc;
	# else: 
	_locp = x.locp;


	if(_locp == -1. or _locp == 1.):
		return pd.Series([frame.addrx,frame.addry]);
	elif(_locp == 2.):
		if(frame.emphours > 0):
			return pd.Series([frame.empx, frame.empy]);
		else:
			#we assert a person who isn't employed
			return pd.Series([frame.addrx, frame.addry]);
	# elif(_locp == 8.):
	# 	if frame.schoollevel == 0. : grid = 3500.0
	# 	if frame.schoollevel == 1. : grid = 1000.0
	# 	if frame.schoollevel == 2. : grid = 2000.0
	# 	if frame.schoollevel == 3. : grid = 3000.0
	# 	elif frame.schoollevel == 4.: grid = 8000.0
	# 	else: grid = 4000.0;
	# 	locx = np.round(frame.addrx / grid,0)*grid;
	# 	locy = np.round(frame.addry / grid,0)*grid;
	# 	return pd.Series([locx,locy]);

	else: 
		if x.prevloc == 2.:
			locx,locy = frame.empx, frame.empy
		else:
			locx,locy = frame.addrx, frame.addry
		
		#take nearest 50 locations and distribute them linearly with a constant probability
		s = list(idx.nearest((locx,locy,locx,locy),50,objects=True))
		pick = np.random.choice(50,p=locprobtab)
		# print(pick, len(s))
		return pd.Series([s[pick].bbox[0],s[pick].bbox[1]])

	# 1 Respondent's home or yard KNOWN
	# 2 Respondent's workplace KNOWN
	# 3 Someone else's home RANDRAD
	# 4 Restaurant or bar RANDRAD
	# 5 Place of worship RANDRAD
	# 6 Grocery store RANDRAD
	# 7 Other store/mall RANDRAD
	# 8 School KNOWN
	# 9 Outdoors away from home RANDRAD
	# 10 Library RANDRAD
	# 11 Other place RANDRAD
	# 30 Bank RANDRAD
	# 31 Gym/health club RANDRAD
	# 32 Post Office RANDRAD
	# 89 Unspecified place RANDRAD
	#
	# 12 Car, truck, or motorcycle (driver) 
	# 13 Car, truck, or motorcycle (passenger)
	# 14 Walking 
	# 15 Bus
	# 16 Subway/train
	# 17 Bicycle 
	# 18 Boat/ferry
	# 19 Taxi/limousine service 
	# 20 Airplane 
	# 21 Other mode of transportation
	# 99 Unspecified mode of transportation
	#
	# else RANDRAD

def daypick(frame,tables,day):
	
	#0 IS 4PM, 1440 is 

	# print(frame)

	weib = tables[0]
	ati = tables[5];
	ita = tables[6];
	itl = tables[7];
	idx = tables[8]

	#determine school/work fixed params
	#????????????????? TODO FIX
	workstart = -1
	if(frame.empshift > -1):
		workstart = pickwork(frame.empshift);
		
	schoolstart = -1
	if(frame.schoollevel > -1):
		schoolstart = pickschool(frame.schoollevel);

	#pick a day type
	dayindex = 2 if (day in (1,7)) else 1;
	cdtable = tables[dayindex][frame.casetype]
	daytype = np.random.choice(len(cdtable),p=cdtable);
	avg = tables[3][daytype][0]
	prior = tables[3][daytype][1]
	locs = tables[3][daytype][2]
	eprior = tables[3][daytype][3]

	

	#determine dependent status and school needs
	#TODO: GOING TO SKIP THIS FOR THE MOMENT

	#select activities and instances for this day, w/expected duration and locations
	actset = collections.Counter();
	for ind,i in enumerate(avg):
		act = ita[ind];
		prob = norm.rvs(loc=i[0],scale=i[1])
		if prob < 0.: prob = 0.0;
		pf,pi = np.modf(prob);
		instances = int(pi + (np.random.rand() < pf))
		actset[act] = min(instances,5);


	df = pd.DataFrame({'act':list(actset.elements())})
	df['actind'] = df['act'].apply(lambda x: ati[x])
	df = df[(df['act']//10000) != 18]
	if(frame.empshift == -1):
		df = df[(df['act']//10000) != 5]


	df['locp'] = df['act'].apply(lambda x: itl[np.random.choice(len(locs[ati[x]]),p=locs[ati[x]])]);
	df['instance'] = df.groupby(['act']).cumcount();


	#df['assign'] = -1;


	#order activities by priority for each instance
	df['pickS'] = norm.cdf(np.random.randn(len(df)));
	df['startguess'] = df.apply(lambda x: np.argmax(prior[x.actind,x.instance] > x.pickS)*5,axis=1)

	df = df.sort_values('startguess')

	df['prior'] = df.apply(lambda x: prior[x.actind,x.instance,int(np.floor(x.startguess/5.))],axis=1)

	df = df.sort_values(['startguess'])


	#df['tlength'] = 0;
	

	#reorder activity sequences w/location preferences

	# I: we know sleep most likely happens at the start and end of the day
	# II: we know that work type activities are likely done in a chunk, so group them
	# III: we know some events interject other events so it's okay if activities are split
	# t = 0.0;

	# df['glength'] = np.abs(df['startguess'].diff(-1).fillna(0.));
	
	
	# df['wlength1'] = np.abs(df['act'].apply(rollweibull,args=(weib,))-df['glength'])
	# df['wlength2'] = np.abs(df['act'].apply(rollweibull,args=(weib,))-df['glength'])
	# df['wlength3'] = np.abs(df['act'].apply(rollweibull,args=(weib,))-df['glength'])
	# df['pickL'] = norm.cdf(np.random.randn(len(df)));
	# df['plength'] = df.apply(lambda x: np.argmax(eprior[x.actind,x.instance] > x.pickL)*5,axis=1)
	# df['plength'] = np.abs( df['plength'].apply(lambda x: 0 if x > 600 else x) -df['glength'])

	# df['avglength'] = np.floor(( df['wlength1']+df['wlength2']+df['wlength3']+df['plength'] ) /4.0)

	# df['minl'] =  df.apply(lambda x: x.avglength if x.avglength < x.glength else x.glength, axis=1)
	# df['maxl'] =  df.apply(lambda x: x.avglength if x.avglength > x.glength else x.glength, axis=1)
	# df['length'] = df.apply(lambda x: np.random.randint(x.minl,x.maxl + 1), axis=1)

	df['wlength'] = df['act'].apply(rollweibull,args=(weib,))
	df['cumlength'] = df['wlength'].cumsum();		
	df['cumlength'] = np.floor(df['cumlength'] / (df['cumlength'].max() / 1440.0))
	df['length'] = np.abs(df['cumlength'].diff(1).fillna(df['cumlength']))

	df['prevloc'] = df['locp'].shift(1).fillna(-1)

	df['locx'],df['locy'] = zip(*df.apply(locApply,args=(frame,),axis=1));

	# 	indices = df['assign'] == -1
	# 	p = df[indices]['prior'].values
	# 	nextact = np.random.choice(df[indices].index.values, p=p/np.sum(p))
	# 	df['assign'].at[nextact] = i;
	# 	thresh = np.random.rand();
	# 	e = eprior[df.at[nextact,'actind'],df.at[nextact,'instance']]
	# 	pick = np.argmax(e > thresh)
	# 	df['end'].at[nextact] = pick*5.0;

	# 	t += df['end'].at[nextact]

	#if(np.random.rand() < 0.05): print(df)
	# print(df)


	#finalize activity times 


	#build the actlist and apply locations
	#actlist += [(t,act,locx,locy)]	
	#actlist = [];

	#actlist += [(10101, 1440, frame['addrx'],frame['addry'])]

	actlist = df[['length','act','locx','locy']].values;

	#movelist

	return actlist;


outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');
def latlongtrans(x):
	#x.locx,x.locy=pyproj.transform(inproj,outproj,x.locx,x.locy);
    
	x.locx,x.locy=pyproj.transformer.transform(inproj,outproj,x.locx,x.locy);  
	return x

def reversetrans(x):
	#x.locx,x.locy=pyproj.transform(outproj,inproj,x.locx,x.locy);
    
	x.locx,x.locy=pyproj.transformer.transform(outproj,inproj,x.locx,x.locy);
	return x

def latlongtransraw(locx,locy):
	#locx,locy=pyproj.transform(inproj,outproj,locx,locy);
    
	locx,locy=pyproj.transformer.transform(inproj,outproj,locx,locy);
	return locx,locy

def reversetransraw(locx,locy):
	#locx,locy=pyproj.transform(outproj,inproj,locx,locy);
    
	locx,locy=pyproj.transformer.transform(inproj,outproj,locx,locy);
	return locx,locy


def gettrip(lonx1,laty1,lonx2,laty2):
	curlobj = pycurl.Curl()
	data = BytesIO()
	#baseurl = "http://localhost:6114/route/v1/driving/"
	baseurl = "http://0.0.0.0:30606/route/v1/driving/"

	baseopts= "?alternatives=3&overview=full&geometries=geojson&annotations=duration"

	#curlobj.setopt(curlobj.URL, "http://0.0.0.0:30606/route/v1/driving/-111.850805,40.767031;-111.8808,40.777031?alternatives=5")
	#print(lonx1, laty1, lonx2, laty2)
	curlobj.setopt(curlobj.URL, baseurl +str(lonx1)+","+str(laty1)+";"+str(lonx2)+","+str(laty2)+baseopts);
	curlobj.setopt(curlobj.WRITEFUNCTION, data.write)
	curlobj.perform()
	# print(data.getvalue());
	dc = json.loads(data.getvalue())
	print(dc);
	if('routes' in dc):
		rcount = len(dc['routes']);
		pick = np.random.randint(0,rcount);
		# print(len(dc['routes'][pick]['geometry']['coordinates']), len(dc['routes'][pick]['legs'][0]['annotation']['distance']))
		line = np.array(dc['routes'][pick]['geometry']['coordinates']).T
		duration = np.array(dc['routes'][pick]['legs'][0]['annotation']['duration'] + [0.0])
		dur = np.sum(duration)
		
		df = pd.DataFrame({"locx":line[0],"locy":line[1],"length":duration});
		df['length'] = (df['length']/60.0)

		df[['locx','locy']] = df[['locx','locy']].apply(reversetrans, axis=1);
		df['triporder'] = np.arange(0,len(df));
		return df,dur;
	else:
		print("ERROR DETECTED IN ROUTE: dumping json");
		print(str(data.getvalue()))
		return None,None;

def gettrip2(lonx1,laty1,lonx2,laty2):
	client = osrm.Client(host='http://localhost:5000')
	response = client.route(coordinates=[[-74.0056, 40.6197], [-74.0034, 40.6333]], overview=osrm.overview.full)

	print(response)

def mangletrips(fr,frame):

	if(len(fr) < 2): return fr;
	
	#trips = fr['locp'].isin([12,13,14,15,16,17,18,19,20,21,99]).index;
	# trips = fr[fr['actind'] < 382][fr['actind'] > 313].index
	# trips = fr[fr['locp'] != fr['prevloc']].index
	fintr = pd.DataFrame();


	for ind in range(1,len(fr)):
		ploc = ind - 1
		if ploc < 0:
			locpx, locpy = frame.addrx, frame.addry;
		else:
			locpx, locpy = fr.iloc[ploc][['locx','locy']]

		# nloc = fr.index.get_loc(ind) #+1
		# if nloc >= len(fr):
		# 	locnx, locny = frame.addrx, frame.addry;
		# else:
		locnx, locny = fr.iloc[ind][['locx','locy']]

		#FIXME: probably need to have a minimum distance limiter here to prevent weird micro trips
		#how to guarantee that a trip has proper road egress (e.g., how to prevent walking off of a freeway?)
		if (locpx == locnx) and (locpy == locny):
			continue;

		# fr.loc[ind]['locx'] = locpx;
		# fr.loc[ind]['locy'] = locpy;
		act = fr.iloc[ind]['actind']
		# low = fr.loc[ind]['start']
		# high = fr.loc[ind]['end']
		# pr = fr.loc[ind]['prevloc']
		loc = fr.iloc[ind]['locp']

		locpx,locpy = latlongtransraw(locpx,locpy);
		locnx,locny = latlongtransraw(locnx,locny);

		trdf,dur = gettrip(locpx,locpy,locnx,locny);
		if(type(trdf) != type(None)):

			trdf['actind']=381; #FIXME: need a better coding system here
			trdf['actorder']=fr.iloc[ind]['actorder']-1;
			# trdf['start']=trdf['length'].cumsum()+low;
			# trdf['end']=trdf['start']+trdf['length']
			# trdf['start']=trdf['start'].apply(lambda x: 1440.0 if x > 1440.0 else x)
			# trdf['end']=trdf['end'].apply(lambda x: 1440.0 if x > 1440.0 else x)
			# trdf['locp']=loc;
			# trdf['prevloc']=pr;
			# trdf.drop(['dur'],axis=1,inplace=True);
			fintr = fintr.append(trdf, ignore_index=True);
		
		# print(trdf)
		# print(fr)
		
	fr = fr.append(fintr, ignore_index=True)

	# frame['lweight'] = frame['lstd'] / frame['lstd'].sum();
	# diff = np.abs(1440.0 - frame['length'].sum()) 

	# actlist['length'] += np.floor(actlist['lweight'] * diff).fillna(1.0);

	return fr;


def picklen(x, lens, jprob):
	# print(x,jprob[x]);
	win = jprob[x].sample(n=1,weights=jprob[x]).index[0]	
	out = lens.iloc[win][['lmin','lmax','lavg','lstd','lhist']]
	return out;


def precsort(actind,precede):
	actlen = len(actind);
	
	pmat = np.random.rand(actlen,actlen);
	omat = np.zeros((actlen,actlen));

	for i in range(actlen):
		for j in range(i+1,actlen):
			result = pmat[i,j] < precede[actind[i],actind[j]]
			omat[i,j] = (1.0 if result else 0.0)
			omat[j,i] = (0.0 if result else 1.0)
	
	return np.sum(omat, axis=0);


def buildseqv2(frame,wins,lens,jointprob,precede,whereprob, dropind,idx):
	#start,end,length, actind
	# print(frame)
	winlen = len(wins)

	
	#FIXME: need to take density into account

	# actlist = wins[np.random.rand(winlen) < wins['prob'].values].copy(deep=True);
	#pick non-picked elements in the non-covered region with accompanying lengths up to three times until coverage is > 0.95

	

	actlist = wins.copy(deep=True);

	# print(actlist)

		# covfrac = np.sum(coverage)/1440.0
		# print("Covfrac: ", i, " ", covfrac)
		# if(covfrac < 0.95):
		# 	minpoint = np.argmin(coverage);
		# 	maxpoint = 1440 - np.argmin(coverage[::-1])
		# 	#  [np.random.rand(winlen) < wins['prob'].values]
		# 	winselect = (wins['wmax'] >= minpoint) & (wins['wmin'] <= maxpoint) & np.invert(allpicks)
			
		# 	picks += [winselect & (np.random.rand(winlen) < wins['prob'].values)]
		# 	allpicks = allpicks | picks[i]
		# 	print(minpoint, maxpoint)
		# else:
		# 	break;
		
		

	try:
		actlist[['lmin','lmax','lavg','lstd','lhist']] = actlist.index.to_series().apply(picklen, args=(lens,jointprob));	
	except KeyError:
		print("There was a keyerror on this iteration: ")
		print(actlist, jointprob, lens);
		return None;
		

		

	# 	for ind,df in actlist[i].iterrows():
	# 		coverage[df.wmin:(df.wmin+df.lmax)] = 1.0

	# coverage = np.reshape(coverage,(48,30))
	# covdist = np.sum(coverage,axis=1)/30
	# print(covdist)

	# covfrac = np.sum(coverage)/1440.0
	# print("Covfrac: ", i, " ", covfrac)
	# print(actlist)

	# actlist = pd.concat(actlist);
	

	# print(actlist)
	
	#ctlist['length'] = actlist.apply(lambda x: x.lstd * np.random.randn() + x.lavg, axis=1)

	actlist['length'] = actlist['lhist'].apply(lambda x: np.random.choice(x[1],p=x[0])).fillna(1.0);
	actlist['length'] = actlist['length'].apply(lambda x: (1.0 if x < 1.0 else np.floor(x)));
	
	#three tries to get enough lmax
	# for i in range(0,3):
	# 	actlist['picked'] = np.maximum(np.random.rand(winlen), 
		

	actind = np.array(actlist.index);
	actlist['precscore'] = precsort(actind,precede)
	actlist = actlist.sort_values(['precscore','wavg','wmin','length','wmax',]);
	

	coverage = np.zeros((1440,))
	actlist['picks'] = False
	actlist['validwin'] = True
	# for i in range(len(actlist)):
	for i in range(3):

		actlist['picks'] = actlist['picks'] | ((np.random.rand(len(actlist)) < actlist['prob'].values) & actlist['validwin']);


		actlist['end'] = (actlist['length']*actlist['picks']).cumsum();

		actlist['start'] = (actlist['end']*actlist['picks']) - actlist['length']
		actlist['start'] = actlist.apply(lambda x: x.end if x.start < 0.0 else x.start, axis=1)

		actlist['validwin'] = actlist.apply(lambda x: True if x.start <= x.wmax and x.start >= x.wmin else False,axis=1)

		if( (actlist['lmax']*actlist['picks']).sum() > 1440):
			break;

	# print(actlist)
	actlist = actlist[actlist['picks']]

	actlist['actorder'] = np.arange(0,len(actlist)*2,2);


	actlist['locp'] = actlist.index.to_series().apply(lambda x: whereprob[x].sample(n=1,weights=whereprob[x]).index[0]).fillna(-1);
	actlist['prevloc'] = actlist['locp'].shift(1).fillna(-1);


	actlist[['locx','locy']] = actlist.apply(locApply,args=(frame,idx),axis=1);
	actlist['triporder'] = np.nan;

	actlist = mangletrips(actlist,frame);


	actlist = actlist.sort_values(['actorder','triporder']).reset_index(drop=True);



	#this is probably bad; does this work right?
	# actlist['lweight'] = actlist['lstd'] / actlist['lstd'].sum();
	# diff = 1440 - actlist['length'].sum() 

	# actlist['length'] += np.floor(actlist['lweight'] * diff).fillna(1.0);

	#CUMSUM IS WRONG; need window smearing that puts activities in their windows correctly

	# actlist['start'] = actlist.apply(lambda x: x.wmin if x.start < x.wmin else x.start  , axis=1)
	# actlist['end'] = actlist['start'] + actlist['length']

	delta = 1440.0 - actlist['length'].sum()
	# print("delta: ", delta);
	actlist['lmaxsmear'] = actlist['lmax'] - actlist['length']
	# actlist['lminsmear'] = actlist['length'] - actlist['lmin']
	# actlist['wminerror'] = np.abs(np.minimum(0.0,actlist['start'] - actlist['wmin'])) #haha bad joke	
	# actlist['wmaxerror'] = np.minimum(0.0,actlist['wmax'] - actlist['start'])

	actlist['lweight'] = actlist['lmaxsmear'] / actlist['lmaxsmear'].sum();
	actlist['lweight'] = actlist.apply(lambda x: x.lweight if x.density > 0.0 else 0.0,axis=1)

	actlist['length'] += np.floor(actlist['lweight'] * delta).fillna(1.0);
	# actlist['validwin'] = actlist.apply(lambda x: True if x.start <= x.wmax and x.start >= x.wmin else False,axis=1)
	# actlist['validlen'] = actlist.apply(lambda x: True if x.length <= x.lmax and x.length >= x.lmin else False, axis=1);

	actlist['end'] = actlist['length'].cumsum();
	actlist['start'] = actlist['end'] - actlist['length']
	
	
	
	# actlist.iloc[len(actlist)-1]['end'] = 1439.0
	# print("sum lmaxsmear: ",actlist['lmaxsmear'].sum())
	# print("sum lminsmear: ",actlist['lminsmear'].sum())
	# print("sum length: ",actlist['length'].sum())
	# print(actlist[actlist['density'] >= 0.0])
	# print("window accuracy: ", actlist[actlist['density'] >= 0.0]['validwin'].sum()/len(actlist[actlist['density'] >= 0.0]))
	# print(actlist[actlist['density'] >= 0.0][['actind','start','length','lweight','locp','validwin','validlen']])




	# print(actlist[["actind","actorder","locp","locx","locy"]])

	actlist.drop(['wincount','winuniq','density','ref','prob','wmin','wmax','wavg','wstd','lmin','lmax','lavg','lstd','precscore','validwin','lhist','triporder','actorder','locp','prevloc','lmaxsmear','lweight','picks'],axis=1,inplace=True) #lweight


	return actlist;


def manageseq(frame,tables,day,idx):

	# ita = tables[6];

	dayindex = 2 if (day in (1,7)) else 1;
	cdtable = tables[dayindex][int(frame.casetype)]
	daytype = np.random.choice(len(cdtable),p=cdtable);
	# wins = tables[3][daytype][0]
	# lens = tables[3][daytype][1]
	# jointprob = tables[3][daytype][2]
	# precede = tables[3][daytype][3]
	# whereprob = tables[3][daytype][4]

	# fr = ad.buildseqv2(wins,lens,jointprob,precede,whereprob);
	fr = buildseqv2(frame,tables[3][daytype][0],tables[3][daytype][1],tables[3][daytype][2],tables[3][daytype][3],tables[3][daytype][4], tables[9],idx);
	if(type(fr) == "NoneType"):
		return nonmobile(frame);

	# fr['prevloc'] = fr['locp'].shift(1).fillna(-1);




	fr['agentnum']=frame.id;
	
	# print(fr);

	# fr.drop(['locp','prevloc'],axis=1,inplace=True);

	

	return fr;


def superfunc(frame, tables, day,idx):
	#mobile or non-mobile?
	if frame['mobile'] == False:
		return nonmobile(frame)
	else:
		return manageseq(frame, tables, day,idx);
		#OLD STUFF
		# return daypick(frame,tables,day);
		#OLDER STUFF
		# if frame['age'] > 18:
		# 	return adult(frame, tables, day);
		# else:
		# 	if frame['age'] > 6:
		# 		return indchild(frame, tables, day);
		# 	else:
		# 		return depchild(frame, tables, day);

#takes the superfunction and gets the horuly grid location activity profile of the user 
def gridsum( frame, grid, tables, day,idx):
	

	# superout = pd.DataFrame(superfunc(frame,tables, day))

	# superout.columns = ['t','act','locx','locy']

	superout = superfunc(frame,tables,day,idx);

	superout['actcode']=superout['actind'].apply(lambda x: tables[6][x]);

	#align
	# superout['locxg'] = superout['locx'].apply(lambda x: np.round(x / grid, 0) )
	# superout['locyg'] = superout['locy'].apply(lambda x: np.round(x / grid, 0) )
	
	# ymin = superout['locyg'].min()
	# ymax = superout['locyg'].max()
	# xmin = superout['locxg'].min()
	# xmax = superout['locxg'].max()
	
	#zero
	# superout['locxg'] = superout['locxg'].apply(lambda x: (x - xmin) )
	# superout['locyg'] = superout['locyg'].apply(lambda x: (x - ymin) )
		
	superout.drop(['end','actind'],axis=1,inplace=True);

	superout['day'] = day;
	superout['day365']=1;
	superout[["locx","locy"]] = superout[["locx","locy"]].apply(latlongtrans,axis=1);
	superout.rename(index=str,columns={"locx":"long","locy":"lat"},inplace=True);
		
	return superout;

#@profile
def parallelapplyfunc(splittable, grid, tables, day, outpath, ttt):
	
	#tpgroup = transprob.groupby(['day','hour']);
	
	idx = buildloctree(tables[8])

	if(len(splittable) > 0):

		supertraj = []

		memcount = 0; 
		print("size: ", splittable.shape[0])
		#print('s',superor,supershape)
		
		con = sqlite3.connect(outpath + '/Ftraj'+str(day)+ttt+"-"+str(mp.current_process().name)+'.sqlite3');

		for i in range(0,splittable.shape[0]):
			traj = gridsum(splittable.iloc[i], grid, tables, day,idx);
			memcount += traj.memory_usage(index=True).sum()
			#supertraj += [traj]

			traj.to_sql('acttraj',con,if_exists='append');

			if(i % 1000 == 0):
				print("PID ",mp.current_process().pid,", step: ",i,", mem = ", memcount/1024/1024);
				sys.stdout.flush();
			
		# supertraj = pd.concat(supertraj,ignore_index=True,axis=0)
		


		# traj.drop(['locx','locy'],axis=1,inplace=True);

		
		
		con.close();


		return None;
		#return supertraj;	

def parallelapplydist(threads, table, grid, tables, day, outpath, ttt):
	#split table
	
	splittable = []
	tsize = len(table)
	tsplit = tsize / threads;
	
	for i in range(threads):
		s = tsplit * i
		e = s + tsplit;
		if e > tsize: e = tsize;
		st = table.iloc[int(s):int(e)]
		#splittable += [(table.iloc[int(s):int(e)],grid,tpgroup,weibull,day)];
		splittable += [(st,grid,tables,day,outpath,ttt)]
	
	p = mp.Pool(threads);
	
	out = p.starmap(parallelapplyfunc,splittable);
	
	# if(len(out) > 0):

	# 	supertraj = []
	# 	for i in range(0,len(out)):
	# 		traj = out[i]
	# 		supertraj += [traj]

	# 	supertraj = pd.concat(supertraj,ignore_index=True,axis=0)	
	# 	return supertraj;	
	# else:
	# 	return None;
	return None;
	
#@profile
def runit(threads):
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	

	print("Threads:",threads)

	limiter = ""
	# limiter = " limit 1000";
	
	print("loading...")

	#NEED TO REDUCE MEMORY AND STUFF
	con = sqlite3.connect(datapath + "indvs2.sq3");
	indvs = pd.read_sql("select * from indvs" + limiter, con);
	keys = ['age','gender','householder','group','mobile']
	for i in keys: indvs[i] = indvs[i].astype('uint8');
	keys = ['id','g1','spouse','g2','household']
	for i in keys: indvs[i] = indvs[i].astype('uint32');
	indvs = indvs.drop('addrn',axis=1)
	indvs = indvs.drop('city',axis=1)
	#age,g1,g2,gender,household,householder,group,mobile,block,addrx,addry,addrn,city,id,spouse
	con.close();
	con = sqlite3.connect(datapath + "employ2.sq3");
	employ = pd.read_sql("select * from employ" + limiter, con);
	#id,empblock,empx,empy,empcode,emphours,empweeks,empshift,probemploy
	keys = ['id']
	for i in keys: employ[i] = employ[i].astype('uint32');
	keys = ['empcode','empweeks','empshift']
	for i in keys: employ[i] = employ[i].astype('int8');
	con.close();
	con = sqlite3.connect(datapath + "school2.sq3");
	school = pd.read_sql("select * from school" + limiter, con);
	#id,schoolprob,schoollevel
	keys = ['schoollevel']
	for i in keys: school[i] = school[i].astype('int8');
	con.close();
	
	#indvlabel = pd.read_csv(datapath + "indvlabels1000.csv",index_col=0)
	indvlabel = pd.read_csv(datapath + "indvlabels.csv",index_col=0)
	
	
	ptable = pd.merge(indvs,employ,on='id')
	ptable = pd.merge(ptable,school,on='id')
	ptable = pd.merge(ptable,indvlabel,on='id')
	ptable = ptable.drop(['index','index_x','index_y'],axis=1)
	ptable.info();
	# con = sqlite3.connect(datapath + "transprob.sq3");
	# transprob = pd.read_sql("select * from transprob", con);
	# con.close();
	
	#tpgroup = transprob.groupby(['day','hour']);
	
	randlocs = pd.read_csv(datapath + "randlocs.csv");
	# idx = buildloctree(randlocs);
	print("randloc len: ", len(randlocs));

	con = sqlite3.connect(datapath + "weibull.sq3");
	weibull = pd.read_sql("select * from weibull", con);
	con.close();
	weibull = weibull.set_index('ID');
	

	labtabfile = h5py.File(datapath + "final-label-classifier/labeltab.h5")
	weekdayarr = labtabfile['/weekday'][:]
	weekendarr = labtabfile['/weekend'][:]
	labtabfile.close();
	

	prioritytab = {}

	

	prior = h5py.File(datapath + "actwindows.h5",'r');
	labels = prior["/labels"][:]
	actmapping = prior["/actmapping"][:];
	locmapping = prior["/locmapping"][:];
	prior.close();
	print(labels)
	dropind = [i for i,k in enumerate(actmapping) if k >= 180000 and k < 189999]

	for i in labels:
		g = "/label-"+str(i)
		wins = pd.read_hdf(datapath + "actwindows.h5",key=g+"/windows");
		lens = pd.read_hdf(datapath + "actwindows.h5",key=g+"/lengthwin");
		jointprob = pd.read_hdf(datapath + "actwindows.h5",key=g+"/jointprob");
		precede = pd.read_hdf(datapath + "actwindows.h5",key=g+"/precede").values;
		whereprob = pd.read_hdf(datapath + "actwindows.h5",key=g+"/whereprob");

		wins = wins.drop(wins[wins['actind'].apply(lambda x:x in dropind)].index);

		# avg = prior[g+"/avginstances"][:];
		# pri = prior[g+"/priorities"][:];
		# epri = prior[g+"/epriorities"][:];
		# loc = prior[g+"/locations"][:];
		# for j in range(len(loc)):
		# 	loc[j] = loc[j] / np.sum(loc[j])
		prioritytab[i] = (wins,lens,jointprob,precede,whereprob);
	# prior.close();
	print([(i,k) for i,k in enumerate(actmapping)]);
	ati = { tr:i for i,tr in enumerate(actmapping) }
	ita = { i:tr for i,tr in enumerate(actmapping) }

	itl = { i:tr for i,tr in enumerate(locmapping) }

	

	commontables = (weibull, weekdayarr, weekendarr, prioritytab, actmapping, ati, ita,  itl, randlocs, dropind )



	print("processing...")

	ttt = time.strftime("-%Y-%m-%d_%H-%M-%S")

	print(datetime.datetime.now().time().isoformat());
	#rawframe = ptable.iloc[1:10000].apply(gridsum, axis=1, args=(500.0,tpgroup, weibull,3,) );
	
	day = int(sys.argv[1])
	print(day)

	# rawmat,x,y,traj = parallelapplyfunc(ptable.iloc[0:1000], 500.0,commontables, day )

	# outpath = "/scratch/kingspeak/serial/u0403692/"
	outpath = "/scratch/general/lustre/u1264235/"
	#for day in range(1,8):
    
	grid = 500.0
	#traj = parallelapplydist(threads, ptable, grid,commontables,day,outpath,ttt )
    
	# traj['day'] = day;
	# traj['day365']=1;
	# traj[["locx","locy"]] = traj[["locx","locy"]].apply(latlongtrans,axis=1);
	# traj.rename(index=str,columns={"locx":"long","locy":"lat"},inplace=True);
	# # traj.drop(['locx','locy'],axis=1,inplace=True);

	# con = sqlite3.connect(datapath + '/Ftraj'+str(day)+ttt+'.sqlite3');
	# traj.to_sql('acttraj',con);
	# con.close();

	print(datetime.datetime.now().time().isoformat());
	

	
	exit();
# 	plt.matshow(rawmat[10,0])
# 	plt.show()
# 	plt.matshow(rawmat[11,0])
# 	plt.show()
# 	plt.matshow(rawmat[12,0])
# 	plt.show()



if __name__ == '__main__':
	threads1 = mkl.get_max_threads();
	print("threads: ", threads1)
	threads = mp.cpu_count();
	print("mp cpu: ", threads)    
	# threads = 5;
	runit(threads)

