import pandas as pd;
import numpy as np;

def columnStrip(frame, columns):
	framecolumns = [col for col in frame.columns if col in columns];
	subframe = frame[framecolumns].copy();
	return subframe;

def vectorizeActs(acttable,demotable,actmapping,demokey,min_step=3):

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