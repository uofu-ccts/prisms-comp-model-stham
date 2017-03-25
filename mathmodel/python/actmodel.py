import pandas as pd;
import numpy as np;
import sqlite3;
import collections;
from scipy.stats import exponweib;
import datetime
import time;
import multiprocessing as mp;
import matplotlib.pyplot as plt;
import h5py;
import mkl;
import sys;



#wac = pd.read_csv(datapath + "lehd/ut_wac_S000_JT00_2014.csv")




#START CLASSIFICATION FUNCTIONS

def nonmobile(x):
	#single activity, 
	return [ (10101, 1440, x['addrx'],x['addry']) ];


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

def daypick(frame,tables,day):
	
	#0 IS 4PM, 1440 is 

	#actlist += [(t,act,locx,locy)]	
	actlist = [];

	#determine school/work fixed params
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
	
	#determine dependent status and school needs
	#TODO: GOING TO SKIP THIS FOR THE MOMENT

	#select mandatory and common acts from sched table w/locs
	

	#build activity sequence using weibull for variance



	actlist += [(10101, 1440, frame['addrx'],frame['addry'])]
	return actlist;


def superfunc(frame, tables, day):
	#mobile or non-mobile?
	if frame['mobile'] == False:
		return nonmobile(frame)
	else:
		return daypick(frame,tables,day);
		#OLD STUFF
		# if frame['age'] > 18:
		# 	return adult(frame, tables, day);
		# else:
		# 	if frame['age'] > 6:
		# 		return indchild(frame, tables, day);
		# 	else:
		# 		return depchild(frame, tables, day);

#takes the superfunction and gets the horuly grid location activity profile of the user 
def gridsum( frame, grid, tables, day):
	
	superout = pd.DataFrame(superfunc(frame,tables, day));

	superout.columns = ['t','act','locx','locy']
	#align
	superout['locx'] = superout['locx'].apply(lambda x: np.round(x / grid, 0) )
	superout['locy'] = superout['locy'].apply(lambda x: np.round(x / grid, 0) )
	
	ymin = superout['locy'].min()
	ymax = superout['locy'].max()
	xmin = superout['locx'].min()
	xmax = superout['locx'].max()
	
	#zero
	superout['locx'] = superout['locx'].apply(lambda x: (x - xmin) )
	superout['locy'] = superout['locy'].apply(lambda x: (x - ymin) )
	
	#print(superout)
	#print(xmin,xmax,ymin,ymax);

	mat = np.zeros((24,3,(xmax-xmin)+1,(ymax-ymin)+1), dtype=np.uint32)
	
	ct = 0;
	for i in range(len(superout)):
		f = superout.iloc[i];
		t = f['t']
		
		sind = int(np.floor(ct/60.0))
		delta = int(np.floor(t/60.0))
		if(sind+delta) > 24:
			delta = 24 - sind;
		
		act = 0;
		if(f['act'] >= 50000 and f['act'] <= 59999):
			act = 1;
		#being at school counts as being at work in this iteration
		if(f['act'] >= 60000 and f['act'] <= 60299):
			act = 1;
		elif(f['act'] >= 180000 and f['act'] <= 189999):
			act = 2;
		
		#act 2 is complicated because we have a range of locations
		#eg, loc1 -> loc2, all taxicab locations possible
		#so we have to slice the locations
		if(act == 2):
			#don't know where the next location is so assume we stay in grid
			if (i+1) >= len(superout):
				nextf = f;
			else:
				nextf = superout.iloc[i+1]
			for j in range(sind,sind+delta):
				for x in np.arange(f['locx'], nextf['locx'],1.0):
					for y in np.arange(f['locy'], nextf['locy'],1.0):
						mat[ j,act,x,y ] += 1
		else:
			for j in range(sind,sind+delta):
				mat[ j,act,f['locx'],f['locy'] ] += 1
		
		ct += t
		if(ct >= 1440):
			break;	
		
	
	
	return mat, xmin, ymin;

#@profile
def parallelapplyfunc(splittable, grid, tables, day):
	
	#tpgroup = transprob.groupby(['day','hour']);
	
	if(len(splittable) > 0):
		supermat,xmin,ymin = gridsum(splittable.iloc[0], grid, tables, day);
		tshape = supermat.shape; #(24,3,x,y)
		supershape = np.array([tshape[2],tshape[3]])
		superor = np.array([xmin,ymin]);
		
		#print('s',superor,supershape)
		
		for i in range(1,splittable.shape[0]):
			mat, xmin, ymin = gridsum(splittable.iloc[i], grid, tables, day);
			tshape = mat.shape; #(24,3,x,y)
			matshape = np.array([tshape[2],tshape[3]])
			mator = np.array([xmin,ymin])
			newor = np.minimum(superor,mator);
			newmax = np.maximum(superor+supershape,mator+matshape)
			newshape = newmax-newor;
			#reshape the matrix if needed
			#print('s',superor,supershape+superor, 'm', mator,matshape+mator, 'n',newor,newshape+newor)
			if( np.max(newshape - supershape) > 0 ):
				#print("reshape")
				newmat = np.zeros((24,3,newshape[0],newshape[1]), dtype=np.uint32)
				loc = superor - newor
				locmax = loc + supershape
				newmat[:,:,loc[0]:locmax[0],loc[1]:locmax[1]] = supermat
				supermat = newmat;
				supershape = newshape
				superor = newor
			#add the new matrix
			loc = mator - superor
			locmax = loc + matshape
			supermat[:,:,loc[0]:locmax[0],loc[1]:locmax[1]] += mat
			
			
		return supermat, superor[0], superor[1];	

def parallelapplydist(threads, table, grid, tables, day):
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
		splittable += [(st,grid,tables,day)]
	
	p = mp.Pool(threads);
	
	out = p.starmap(parallelapplyfunc,splittable);
	
	if(len(out) > 0):
		supermat,xmin,ymin = out[0]
		tshape = supermat.shape; #(24,3,x,y)
		
		supershape = np.array([tshape[2],tshape[3]])

		superor = np.array([xmin,ymin]);
		
		for i in range(1,len(out)):
			mat, xmin, ymin = out[i]
			tshape = mat.shape; #(24,3,x,y)
			matshape = np.array([tshape[2],tshape[3]])
			mator = np.array([xmin,ymin])
			newor = np.minimum(superor,mator);
			newmax = np.maximum(superor+supershape,mator+matshape)
			newshape = newmax-newor;
			#reshape the matrix if needed
			if( np.max(newshape - supershape) > 0 ):
				#print("reshape")
				newmat = np.zeros((24,3,newshape[0],newshape[1]), dtype=np.uint32)
				loc = superor - newor
				locmax = loc + supershape
				newmat[:,:,loc[0]:locmax[0],loc[1]:locmax[1]] = supermat
				supermat = newmat;
				supershape = newshape
				superor = newor
				#print(supermat.shape)
			#add the new matrix
			loc = mator - superor
			locmax = loc + matshape
			supermat[:,:,loc[0]:locmax[0],loc[1]:locmax[1]] += mat
			
		return supermat, superor[0], superor[1];	
	else:
		return None;
	
#@profile
def runit(threads):
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	
	limiter = " limit 1000";

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
	keys = ['id','emphours']
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
	
	indvlabel = pd.read_csv(datapath + "indvlabels1000.csv",index_col=0)
	
	
	ptable = pd.merge(indvs,employ,on='id')
	ptable = pd.merge(ptable,school,on='id')
	ptable = pd.merge(ptable,indvlabel,on='id')
	ptable = ptable.drop(['index','index_x','index_y'],axis=1)
	ptable.info();
	# con = sqlite3.connect(datapath + "transprob.sq3");
	# transprob = pd.read_sql("select * from transprob", con);
	# con.close();
	
	#tpgroup = transprob.groupby(['day','hour']);
	
	con = sqlite3.connect(datapath + "weibull.sq3");
	weibull = pd.read_sql("select * from weibull", con);
	con.close();
	weibull = weibull.set_index('ID');
	

	labtabfile = h5py.File(datapath + "final-label-classifier/labeltab.h5")
	weekdayarr = labtabfile['/weekday'][:]
	weekendarr = labtabfile['/weekend'][:]
	labtabfile.close();
	

	
	commontables = (weibull, weekdayarr, weekendarr )



	print("processing...")
	print(datetime.datetime.now().time().isoformat());
	#rawframe = ptable.iloc[1:10000].apply(gridsum, axis=1, args=(500.0,tpgroup, weibull,3,) );
	
	day = int(sys.argv[1])
	print(day)

	#rawmat,x,y = parallelapplyfunc(ptable.iloc[0:10000], 500.0,transprob, weibull,3 )
	out = h5py.File(datapath + '/test/Finfluence'+str(day)+time.strftime("-%Y-%m-%d_%H-%M-%S")+'.h5')
	#for day in range(1,8):
	rawmat, x, y = parallelapplydist(threads, ptable, 500.0,commontables,day )
	ds = out.create_dataset('/populations',data=rawmat,fillvalue=0.,compression='gzip',compression_opts=9)
	ds.attrs['xorigin'] = x * 500.0;
	ds.attrs['yorigin'] = y * 500.0;
	ds.attrs['day'] = day;
	ds.attrs['date']=datetime.datetime.now().isoformat() #place holder
	ds.attrs['grid']=500.0
		
	out.close();
	
	
	print(datetime.datetime.now().time().isoformat());
	print(x,y);
	print(rawmat.shape);
	exit();
# 	plt.matshow(rawmat[10,0])
# 	plt.show()
# 	plt.matshow(rawmat[11,0])
# 	plt.show()
# 	plt.matshow(rawmat[12,0])
# 	plt.show()



if __name__ == '__main__':
	#threads = mkl.get_max_threads();
	threads = 1;
	runit(threads)


		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

