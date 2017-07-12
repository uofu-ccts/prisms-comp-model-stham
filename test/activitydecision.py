import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
#import scipy.stats.norm as norm;
from scipy.stats import halfnorm, norm;
import numpy as np;
import sys;

def propkernel(start,end,edgeblock):
	edgewidth = len(edgeblock);
	out = np.zeros((1440,));


	out[start:end] += 1.0;
	
	
	sedge = max(start-edgewidth,0);
	slen = min(start, edgewidth);
	eedge = min(1440,(end+edgewidth))
	elen = min((1440-end), edgewidth)

	out[sedge:start] += edgeblock[:slen][::-1];
	out[end:eedge] += edgeblock[:elen];

	return out;

def propkernhalfgauss(sidewidth):
	invals = np.linspace(0.0,3.0,sidewidth);
	out = halfnorm.pdf(invals);
	# print(out/out.max());
	return out/out.max();

def actprop(acttable, actmapping, sidewidth=20, limit=0.25):

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	edgeblock = propkernhalfgauss(sidewidth);

	for i,df in acttable.iterrows():
		out[df.actind] += propkernel(df.start,df.end,edgeblock);
	
	# for i in range(len(out)):
	# 	out[i] = out[i] / (out[i].max()+0.0001);
	# out = out/float(casecount);
	limitcount = limit*casecount
	out = np.clip(out,0.0,limitcount)/limitcount;

	return out;


def nextkernel(start,length):
	
	klength = length*2.0

	kern = norm.pdf(np.linspace(-3.0,3.0,klength));
	out = np.zeros((1440,));

	
	
	sedge = max(0,start-length);
	eedge = min(1440,start+length);
	skern = max(0, length-start);
	ekern = min(klength, klength + 1440 - start)

	# print(start, length ,sedge, eedge, skern, ekern);

	out[sedge:eedge] = kern[skern:ekern]

	return out;

def actnext(acttable, actmapping, sidelen=0.5, limit=0.25):

	actcount = len(actmapping);

	casemap = np.sort(list(set(acttable['TUCASEID'])))
	casecount = len(casemap)

	out = np.zeros((actcount,1440));

	for i,df in acttable.iterrows():
		out[df.actind] += nextkernel(df.start,np.ceil(df.length*sidelen));

	limitcount = limit*casecount
	out = np.clip(out,0.0,limitcount)/limitcount;

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
print([(i,k) for i,k in enumerate(actmapping)]);

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
# print([(i,k) for i,k in enumerate(locmapping)]);

labellist = np.sort(list(set(acttable['daytypelabelreduce'])))

acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
acttable['end'] = acttable['TUCUMDUR24']
acttable['length'] = acttable['TUACTDUR24']
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);



for i,df in acttable.groupby(['daytypelabelreduce']):
	c = 0;
	if (c > 3): break;
	print("calc:",i)
	prop = actnext(df,actmapping,sidelen=0.5,limit=0.15);

	plt.matshow(prop);
	plt.show();
	
	
	c += 1