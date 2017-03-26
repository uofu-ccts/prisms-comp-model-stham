import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
#import scipy.stats.norm as norm;
from scipy.stats import norm;
import numpy as np;

import h5py;


def slide(x,width):
	out = np.zeros_like(x);

	if(width < 1): width = 1;

	for i in range(len(x)):
		first = i - width;
		if(first < 0): first = 0;
		out[i] = np.sum(x[first:i]);

	return out;

def slide2(x,width,count):
	out = np.zeros_like(x);

	#if(width < 1): width = 1;

	val = 0;
	for i in range(len(x)):
		val += x[i];
		out[i] = val/count;
		if(out[i] > width): val = 0;
	return out;


#table to be sorted, width of window, interval size
def slidewindow(table, maxinstances, interval_size,actmapping,locmapping):

	intervals = 1440 / interval_size;
	actcount = len(actmapping);

	casemap = np.sort(list(set(table['TUCASEID'])))
	casecount = len(casemap)

	cri = { tr:i for i,tr in enumerate(casemap) }
	itc = { i:tr for i,tr in enumerate(casemap) }

	#calc basecounts

	basecount = np.zeros((actcount,casecount))
	for i,df in table.iterrows():
		basecount[df.actind,cri[df.TUCASEID]] += 1

	
	
	avginstances = np.zeros((actcount,2));
	for i in range(len(basecount)):
		if(np.sum(basecount[i]) < 1): continue;
		p = norm.fit(basecount[i]);
		# a,b = p;
		avginstances[i] =  p
		# x = np.linspace(-5.,5.,1000);
		# plt.plot(x,norm.pdf(x,a,b));
	
	# print(avginstances);
	# plt.show();
	
	#calc starttimes cohorts
	startcount = np.zeros((actcount,maxinstances, intervals));
	# instancecount = np.zeros((actcount,maxinstances));

	table.sort_values(['actind','TUCASEID','TUACTIVITY_N'],inplace=True)

	table['instance'] = table.groupby(['actind','TUCASEID']).cumcount();

	#print(table[['actind','TUCASEID','TUACTIVITY_N','instance']].iloc[0:100])

	for i,df in table.iterrows():
		if(df.instance >= maxinstances): continue;
		index = np.floor(df.start / interval_size)
		startcount[df.actind,df.instance,index] += 1;
		# instancecount[df.actind,df.instance] += 1;

	
	
	for i in range(len(startcount)):
		for j in range(len(startcount[i])):
			startcount[i,j] = np.cumsum(startcount[i,j]);
			# startcount[i,j] /= np.amax(startcount[i,j]);



	#get the location information
	loccount = len(locmapping);
	locations = np.zeros((actcount,loccount));
	for i,df in table.iterrows():
		locations[df.actind,df.whereind] += 1;

	for i in range(len(locations)):
		locations[i] /= np.sum(locations[i]);
		# plt.plot(locations[i])
	
	# plt.show();

	#compute the slider

	#slidewidth = np.floor(bandwidth / interval_size);

	#convert = np.convolve(verticalsum,np.ones(slidewidth,dtype=int),'same')

	#for i in range(len(startcount)):
		#startcount[i] = np.convolve(startcount[i],np.ones(slidewidth,dtype=int),'same')
		#startcount[i] = slide2(startcount[i],0.95,casecount);
		#startcount[i] /= totalcount;

	return avginstances,startcount,locations;

def plotslidewindowOLD(mat,mapping,title,path):
	
	
	#set up coloration
	start = 0.0
	stop = 1.0
	number_of_lines=19
	cm_subsection = np.linspace(start, stop, number_of_lines)
	maincolors = [ cm.prism(x) for x in cm_subsection ]
	colors = []
	for i in mapping:
		tier1 = i // 10000;
		tier2 = (i // 100) - (tier1 * 100)
		if (tier1 == 50): tier1 = 19;
		if (tier2 == 99): tier2 = 19; 
		tier1 = tier1 - 1;
		tier2 = tier2 - 1
		
		scol = list(maincolors[tier1])
		colors += [tuple(scol)];


	#set up labeling
	legart = []
	leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
	for i in range(number_of_lines):
		legart += [mpatch.Rectangle((0,0),1,1,fc=maincolors[i])]


	intervals = len(mat[0]);

	x = np.linspace(0.0,24.0,intervals)
	
		
	additive = np.zeros(intervals)
	datasum = np.cumsum(mat, axis=0);

	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	#ax.set_ylim([0.0,1.0])
	ax.set_xlim([0.0,24.0])

	ax.fill_between(x,0,datasum[0],facecolor=colors[0],linewidth=0.1)
	for i in range(len(colors)-2):
		ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],linewidth=0.1)

	plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(title)
	#F = plt.gcf();
	#F.set_size_inches(12,8)
	#F.set_dpi(300);
	#F.savefig(prefix +"label-" +str(jg[0]) + ".png");
	plt.show();
	plt.clf();

def plotslidewindow(mat,mapping,title,path):
	
	#set up coloration
	start = 0.0
	stop = 1.0
	number_of_lines=19
	cm_subsection = np.linspace(start, stop, number_of_lines)
	maincolors = [ cm.prism(x) for x in cm_subsection ]
	colors = []
	for i in mapping:
		tier1 = i // 10000;
		tier2 = (i // 100) - (tier1 * 100)
		if (tier1 == 50): tier1 = 19;
		if (tier2 == 99): tier2 = 19; 
		tier1 = tier1 - 1;
		tier2 = tier2 - 1
		
		scol = list(maincolors[tier1])
		colors += [tuple(scol)];


	#set up labeling
	legart = []
	leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
	for i in range(number_of_lines):
		legart += [mpatch.Rectangle((0,0),1,1,fc=maincolors[i])]


	intervals = len(mat[0][0]);

	x = np.linspace(0.0,24.0,intervals)
	
		
	# additive = np.zeros(intervals)
	# datasum = np.cumsum(mat, axis=0);

	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	#ax.set_ylim([0.0,1.0])
	ax.set_xlim([0.0,24.0])

	# ax.fill_between(x,0,datasum[0],facecolor=colors[0],linewidth=0.1)
	# for i in range(len(colors)-2):
	# 	ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],linewidth=0.1)
	for i in range(len(mat)):
		for j in range(len(mat[i])):
		# for j in range(1):	
			ax.plot(x,mat[i,j],color=colors[i])
		# if(i >): break;

	plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title(title)
	#F = plt.gcf();
	#F.set_size_inches(12,8)
	#F.set_dpi(300);
	#F.savefig(prefix +"label-" +str(jg[0]) + ".png");
	plt.show();
	plt.clf();

###############
#    BEGIN    #
###############

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")

labeltab = pd.read_csv(datapath + "final-label-classifier/labels.csv")

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
acttable['actind'] = acttable['TRCODE'].apply(lambda x: ati[x]);
acttable['whereind'] = acttable['TEWHERE'].apply(lambda x: wti[x]);

#print(acttable);

outfile = h5py.File(datapath + "actdata.h5");

outfile.create_dataset("/mapping",data=locmapping,fillvalue=0.,compression='gzip',compression_opts=9)
outfile.create_dataset("/labels",data=labellist,fillvalue=0.,compression='gzip',compression_opts=9)

for i,df in acttable.groupby(['daytypelabelreduce']):
	print(i,end=' ');
	avginstances,priormat,locations = slidewindow(df,5,5,actmapping,locmapping)

	outfile.create_dataset("/label-"+str(i)+"/avginstances",data=avginstances,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.create_dataset("/label-"+str(i)+"/priorities",data=priormat,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.create_dataset("/label-"+str(i)+"/locations",data=locations,fillvalue=0.,compression='gzip',compression_opts=9)

	
	#plotslidewindow(slidemat,actmapping,"label "+str(i),"");

outfile.close();