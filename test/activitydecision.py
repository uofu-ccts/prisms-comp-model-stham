import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
#import scipy.stats.norm as norm;
from scipy.stats import halfnorm;
import numpy as np;
import sys;

def propkernel(start,end,edgeblock):
	edgewidth = len(edgeblock);
	out = np.zeros((1440,));
	out[start:end] = 1.0;
	out[max((start-edgewidth),0):start] = edge;
	out[end:min(1440,(end+edgewidth))] = edge[::-1];

def propkerngauss(sidewidth):
	invals = np.linspace(0.0,3.0,sidewidth);
	out = halfnorm(invals);
	return out/out.max();

def actprop(acttable, actmapping, sidewidth=20):

	actcount = len(actmapping);

	casemap = np.sort(list(set(table['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	edgeblock = propkerngauss(sidewidth);

	for i,df in acttable.iterrows():
		out[df.actind] += propkernel(df.start,df.end,edgeblock);

	out = out/float(casecount);

	return out;


###############
#    BEGIN    #
###############

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")

labeltab = pd.read_csv(datapath + "newclassify-final/labels.csv")

acttable = pd.merge(acttable,labeltab,on="TUCASEID");

acttable = acttable[['TUCASEID','TRCODE','daytypelabelreduce','TEWHERE','TUACTDUR24','TUCUMDUR24','TUACTIVITY_N']]

actmapping = np.sort(list(set(acttable['TRCODE'])))
#actcount = len(mapping)
ati = { tr:i for i,tr in enumerate(actmapping) }
ita = { i:tr for i,tr in enumerate(actmapping) }

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
# print([(i,k) for i,k in enumerate(locmapping)]);

labellist = np.sort(list(set(acttable['daytypelabelreduce'])))

acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
acttable['end'] = acttable['TUACTDUR24']
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);

for i,df in acttable.groupby(['daytypelabelreduce']):
	c = 0;
	if (c > 3): break;

	prop = actprop(acttable,actmapping);

	plt.matshow(prop);
	plt.show();
	plt.clf();
	
	c += 1