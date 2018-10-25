import pandas as pd;
import numpy as np;
import h5py;
import mkl;



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

def mangletrips(fr):

	if(len(fr) < 2): return fr;
	
	fintr = pd.DataFrame({"locp":fr["locp"],"prevloc":fr["prevloc"],"actorder":fr["actorder"]-1})
	fintr["actind"]=381
	#travel time is average 20 min with 5 min standard dev, clipped to [0.0,40.0]
	fintr["length"]=np.clip(np.random.randn(len(fr))*5.0+20.0,0.0,40.0)
		# print(trdf)
		# print(fr)
	fintr["locp"] = fintr["locp"].apply(lambda x: 1 if x==-1 else x);
	fintr["prevloc"] = fintr["prevloc"].apply(lambda x: 1 if x==-1 else x);
	fintr = fintr[fintr["locp"]!=fintr["prevloc"]]
		
	fr = fr.append(fintr, ignore_index=True)

	# frame['lweight'] = frame['lstd'] / frame['lstd'].sum();
	# diff = np.abs(1440.0 - frame['length'].sum()) 

	# actlist['length'] += np.floor(actlist['lweight'] * diff).fillna(1.0);

	return fr;

def buildseqv2(daytype,wins,lens,jointprob,precede,whereprob, dropind):
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

	#FIXME
	# actlist[['locx','locy']] = actlist.apply(locApply,args=(frame,idx),axis=1);
	actlist['triporder'] = np.nan;

	#FIXME
	actlist = mangletrips(actlist);


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
	# actlist.drop(['wincount','winuniq','density','ref','prob','wmin','wmax','wavg','wstd','lmin','lmax','lavg','lstd','precscore','validwin','lhist','triporder','lmaxsmear','lweight','picks'],axis=1,inplace=True) #lweight


	return actlist;



def manageseq(daytype,tables,day):

	# ita = tables[6];

	dayindex = 2 if (day in (1,7)) else 1;
	# cdtable = tables[dayindex][int(frame.casetype)]
	# daytype = np.random.choice(len(cdtable),p=cdtable);
	# wins = tables[3][daytype][0]
	# lens = tables[3][daytype][1]
	# jointprob = tables[3][daytype][2]
	# precede = tables[3][daytype][3]
	# whereprob = tables[3][daytype][4]

	# fr = ad.buildseqv2(wins,lens,jointprob,precede,whereprob);
	fr = buildseqv2(daytype,tables[3][daytype][0],tables[3][daytype][1],tables[3][daytype][2],tables[3][daytype][3],tables[3][daytype][4], tables[9]);


	# fr['prevloc'] = fr['locp'].shift(1).fillna(-1);




	# fr['agentnum']=frame.id;
	
	# print(fr);

	# fr.drop(['locp','prevloc'],axis=1,inplace=True);

	

	return fr;



def runit():

	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

	print("loading...")

	con = sqlite3.connect(datapath + "weibull.sq3");
	weibull = pd.read_sql("select * from weibull", con);
	con.close();
	weibull = weibull.set_index('ID');

	labtabfile = h5py.File(datapath + "final-label-classifier/labeltab.h5",'r')
	weekdayarr = labtabfile['/weekday'][:]
	weekendarr = labtabfile['/weekend'][:]
	labtabfile.close();

	prioritytab = {}
	prior = h5py.File(datapath + "actwindows.h5",'r');
	labels = prior["/labels"][:]
	actmapping = prior["/actmapping"][:];
	locmapping = prior["/locmapping"][:];
	prior.close();

	dropind = [i for i,k in enumerate(actmapping) if k >= 180000 and k < 189999]

	for i in labels:
		g = "/label-"+str(i)
		wins = pd.read_hdf(datapath + "actwindows.h5",key=g+"/windows",mode='r');
		lens = pd.read_hdf(datapath + "actwindows.h5",key=g+"/lengthwin",mode='r');
		jointprob = pd.read_hdf(datapath + "actwindows.h5",key=g+"/jointprob",mode='r');
		precede = pd.read_hdf(datapath + "actwindows.h5",key=g+"/precede",mode='r').values;
		whereprob = pd.read_hdf(datapath + "actwindows.h5",key=g+"/whereprob",mode='r');

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

	#FIXME
	empty = []
	# commontables = (weibull, weekdayarr, weekendarr, prioritytab, actmapping, ati, ita,  itl, randlocs, dropind )
	commontables = (empty, weekdayarr, weekendarr, prioritytab, actmapping, ati, ita,  itl, empty, dropind )



	print("making sims...")
	# print(manageseq(0,commontables,4))

	simseqs = []
	samples = 500
	c = 0;
	for i in labels:
		for j in range(samples):
			seq = manageseq(i,commontables,4)
			seq["agentnum"] = c;
			seq["daytype"] = i
			simseqs += [seq];
			c+=1
	
	print("writing...")
	simseqs = pd.concat(simseqs,ignore_index=True);
	simseqs.to_csv(datapath + "simseqs.csv")


if __name__ == "__main__":
	threads = mkl.get_max_threads();
	runit();