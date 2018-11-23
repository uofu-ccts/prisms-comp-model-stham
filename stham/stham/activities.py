import pandas as pd;
import numpy as np;
from trajectories import PTe,AWe,LWe;
from sklearn.mixture import BayesianGaussianMixture;

def columnStrip(frame, columns):
	framecolumns = [col for col in frame.columns if col in columns];
	subframe = frame[framecolumns].copy();
	return subframe;


#TODO: parallelize
def vectorizeActs(acttable,actmapping,demokey,min_step=3):

	if( min_step < 1 ): min_step = 1
	actsteps = 1440 // min_step
	actcount = len(actmapping)

	acttable['minstart'] = acttable['start'] // min_step;
	acttable['minlength'] = acttable['length'] // min_step;

	acttable['minlength'] = acttable['minlength'].apply(lambda x:1 if x <= 0 else x)

	cases = acttable.groupby(['case']) 
	casecount = len(cases)

	vectorsCount = np.zeros((casecount,actcount))
	vectorsTime = np.zeros((casecount,actsteps))
	vectorsDemo = np.zeros_like(demotable.values);

	for ind,i in enumerate(cases):
		for n,j in i[1].iterrows():
			vectorsCount[ind][j['actcode']] += j['length'];
		
		g,df = i;
		df = df.sort_values(["start"])
		vf = np.zeros((actsteps));
		for row in df.iterrows():
			vf[(row[1]['minstart']):(row[1]['minstart']+row[1]['minlength'])] = row[1]['actcode'];
			
		vectorsTime[ind] = vf;

		vectorsDemo[ind] = demotable[demotable[demokey]==i[1]['case'].iloc[0]].values;

	
	#supercolumns = [str(b) for b in mapping] + list(infotable.columns) ;
	minlist = [("min"+ str(i).zfill(4)) for i in range(0,1440,min_step)]

	supercolumns = [str(b) for b in actmapping] + minlist + list(demotable.columns) ;
	superframe = pd.DataFrame(np.concatenate((vectorsCount,vectorsTime,vectorsDemo),axis=1),columns=supercolumns);
	supervec = superframe.set_index(demotable.index).values;

	return supervec,supercolumns;



def breaks1d(df, ncomp=10):
	mat = df.values;
	# print(matX)
	bgm = BayesianGaussianMixture(n_components=ncomp,covariance_type='full',max_iter=500,n_init=4).fit(mat);
	pred = bgm.predict(mat);

	return pred;

def assignWindow(x,wins):
	out = wins[wins['actind']==x['actind']][wins['wmax']>=x['start']][wins['wmin']<=x['start']].index;
	if(len(out) < 1): return -1;
	return out[0];

def assignLen(x,lens):
	out = lens[lens['actind']==x['actind']][lens['lmax']>=x['length']][lens['lmin']<=x['length']] .index;
	if(len(out) < 1): return -1;
	return out[0];

def phist(x,bins):
	a,b = np.histogram(x['length'].values,bins=bins);
	a = a/np.sum(a)
	return np.array(a),np.array(b);

def getwindows(df,bins=10,ncomp=10):
	ncases = df['case'].unique().size
	# print(ncases,len(df));

	allwindows = pd.DataFrame();
	alllengths = pd.DataFrame();

	for i,g in df.groupby(['actcode']):
		if len(g) < 10: continue;
		g['window'] = breaks1d(g[["start"]]);
		g['lenwin'] = breaks1d(g[["length"]]);

		windows = pd.DataFrame(index=g['window'].unique());
		windows['winuniq'] =  g.groupby(['window','case']).size().reset_index().groupby('window').size();
		windows['actprob'] = windows['winuniq'] / ncases;
		windows['wmin'] = g.groupby('window').apply(lambda x: x['start'].min());
		windows['wmax'] = g.groupby('window').apply(lambda x: x['start'].max());
		windows['wavg'] = g.groupby('window').apply(lambda x: x['start'].mean());
		windows['actcode'] = i;

		allwindows = allwindows.append(windows);

		lengths =pd.DataFrame(index=g['lenwin'].unique());
		lengths['ref'] = lengths.index;
		lengths['lmin'] = g.groupby('lenwin').apply(lambda x: x['length'].min());
		lengths['lmax'] = g.groupby('lenwin').apply(lambda x: x['length'].max());
		lengths['lhist'] = 0; lengths['lbins'] = 0;
		#FIXME when not tired
		result = g.groupby('lenwin').apply(phist,bins);
		print(result)
		lengths['actind'] = i;

		alllengths = alllengths.append(lengths);

	allwindows = allwindows.reset_index();
	allwindows = allwindows.drop("index",axis=1);
	alllengths = alllengths.reset_index();
	alllengths = alllengths.drop("index",axis=1);

	return allwindows,alllengths;

def getPrecedeMat(df,wins):
	
	wincount = len(wins);
	precede = np.zeros((wincount,wincount));

	g = df.groupby(['wins'])
	for i,iwin in wins.iterrows():
		for j,jwin in wins.iterrows():
			if(i==j): continue;
			#non-overlapping windows
			if( iwin['wmax'] < jwin['wmin'] ):
				precede[i,j] = 1.0
			#overlapping windows
			else:
				pcount = 0.0;
				mat = pd.merge(g.get_group(i)[['case','start']],g.get_group(j)[['case','start']],how='outer',on='case')
				count = len(mat)
				mat['start_x'].fillna(mat['start_x'].dropna().sample(n=count,replace=True).reset_index(drop=True),inplace=True)
				mat['start_y'].fillna(mat['start_y'].dropna().sample(n=count,replace=True).reset_index(drop=True),inplace=True)
				pcount = mat.apply(lambda x: np.float(x.start_x < x.start_y),axis=1).sum()
				precede[i,j] = pcount / np.float(count);

	return precede;

#TODO: parallelize at global level
def buildWindow(acttable,lhistbins=10):

	wins, lens = getwindows(acttable,bins=lhistbins);
	#there's a weird interaction here where the window sorting is actually really important
	#the activity could be asasigned to more than one start window
	#but assignment only takes the first window
	#by sorting with descending start time, we guarantee that we always assign a window
	#preventing the issue with the joint probability window lacking an 
	#index for the window
	wins = wins.sort_values(['wmin','prob'],ascending=[False,False]);
	acttable['wins'] = acttable[['actind','start']].apply(assignWindow,args=(wins,),axis=1);
	acttable['lwins'] = acttable[['actind','length']].apply(assignLen,args=(lens,),axis=1);
	acttable = acttable.sort_values(['case','start'])

	lenactjointprob = acttable.groupby(['wins']).apply(lambda x: x['lwins'].value_counts() / x['lwins'].count() ).values;
	# print(jointprob);

	whereprob = acttable.groupby(['wins']).apply(lambda x: x['where'].value_counts() / x['where'].count() ).values;

	precede = getPrecedeMat(df,wins);


	winmat = wins[['actcode','actprob','wmin','wamx','wavg']].values;
	lenmat = lens[['lmin','lmax']].values;
	lhist = np.array(lens["lhist"].values)
	lbins = np.array(lens["lbins"].values)

	return {PTe.ACTCOUNT:len(wins),PTe.LENCOUNT:len(lens),PTe.LENACTJOINTPROB:lenactjointprob,PTe.ACTWINS:winmat,PTe.LENWINS:lenmat,PTe.LHIST:lhist,PTe.LBINS:lbins,PTe.ORDERPROB:precede,PTe.WHEREPROB:whereprob}


