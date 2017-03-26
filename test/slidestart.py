import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;


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
def slidewindow(table, bandwidth, interval_size):

	intervals = 1440 / interval_size;
	actcount = max(list(table['actind'])) + 1
	totalcount = len(set(table['TUCASEID']));
	
	#calc starttimes
	startcount = np.zeros((actcount, intervals));
	verticalsum = np.zeros(intervals);

	for i,df in table.iterrows():
		index = np.floor(df.start / interval_size)
		startcount[df.actind,index] += 1;
		verticalsum[index] += 1;
	
	# for i in range(len(startcount)):
	# 	startcount[i] /= verticalsum;

	#compute the slider

	slidewidth = np.floor(bandwidth / interval_size);

	#convert = np.convolve(verticalsum,np.ones(slidewidth,dtype=int),'same')

	for i in range(len(startcount)):
		#startcount[i] = np.convolve(startcount[i],np.ones(slidewidth,dtype=int),'same')
		startcount[i] = slide2(startcount[i],0.95,totalcount);
		#startcount[i] /= totalcount;

	return startcount;

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


	intervals = len(mat[0]);

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
		ax.plot(x,mat[i],color=colors[i])

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
acttable = acttable[['TUCASEID','TRCODE','daytypelabelreduce','TEWHERE','TUACTDUR24','TUCUMDUR24']]

mapping = np.sort(list(set(acttable['TRCODE'])))
actcount = len(mapping)

tri = { tr:i for i,tr in enumerate(mapping) }
itr = { i:tr for i,tr in enumerate(mapping) }

acttable['start'] = acttable['TUCUMDUR24']-acttable['TUACTDUR24']
acttable['actind'] = acttable['TRCODE'].apply(lambda x: tri[x]);

#print(acttable);


for i,df in acttable.groupby(['daytypelabelreduce']):

	slidemat = slidewindow(df,180,5)
	plotslidewindow(slidemat,mapping,"label "+str(i),"");

