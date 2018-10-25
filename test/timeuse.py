import pandas as pd;
import time;
import matplotlib
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;
from collections import Counter;

# sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/test")
# import bettersankey as bsk;

matplotlib.rcParams['hatch.linewidth']=0.5

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

print("reading...")
acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "timeuse/atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')

labels = pd.read_csv(datapath + "final-label-classifier/labels.csv")

acttable = pd.merge(acttable,labels[['daytypelabelreduce','TUCASEID','TUDIARYDAY']],on='TUCASEID')


# jointable.groupby(['TUDIARYDAY','TUTIER1CODE','TUTIER2CODE']).count().TUCASEID.plot(kind='bar')
# plt.show()
# 
# exit()
#for each activity record find the 
def demoActPlot(frame,labelcolumn,cutoff=0):
		

	weekdayset = [2,3,4,5,6]
	#prelim stuff
	# mapping = np.sort(list(set(frame['TRCODE'])))
	mapping = np.sort(list(set(frame['TUTIER1CODE'])))
	actcount = len(mapping)
	actcount = 7
	# tri = 

	print(mapping)

	tri = { tr:i for i,tr in enumerate(mapping) }
	#reduced mapping for publication
	# tri = {1: 0, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4, 7: 6, 8: 6, 9: 6, 10: 6, 11: 10, 12: 11, 13: 11, 14: 11, 15: 11, 16: 11, 18: 16, 50: 11}
	# tri = {1: 0, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4, 7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 6, 13: 6, 14: 6, 15: 6, 16: 6, 18: 5, 50: 6}
	tri = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 4, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 18: 6, 50: 5}
	print(tri)
	
	#set up coloration
	start = 0.25
	stop = 1.0
	number_of_lines=7
	# cms = np.linspace(start, stop, number_of_lines)
	cms = np.linspace(stop, start, number_of_lines)
	# cms = [ p[0],p[4],p[1],p[5],p[2],p[6],p[3] ]
	hatchset = [ '', '////' , '\\\\\\\\', '||||' , '----' , '....' , '' , 'o' , 'O' , '.' , '*' ]
	# maincolors = [ cm.plasma(x) for x in cms ]
	maincolors = [ cm.viridis(x) for x in cms ]
	colors = maincolors

	# colors = []
	# for i in mapping:
	# 	tier1 = i // 10000;
	# 	# tier2 = (i // 100) - (tier1 * 100)
	# 	if (tier1 == 50): tier1 = 19;
	# 	# if (tier2 == 99): tier2 = 19; 
	# 	tier1 = tier1 - 1;
	# 	# tier2 = tier2 - 1
		
	# 	scol = list(maincolors[tier1])
	# # 	scol[2] = scol[2] + (tier2 * 0.1)
	# # 	if(scol[2] > 1.0): scol[2] = 1.0;
	# # 	scol[3] = scol[3] + (tier2 * 0.1)
	# # 	if(scol[3] > 1.0): scol[3] = 1.0;
	# 	colors += [tuple(scol)];


	#set up labeling
	legart = []
	# leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
	leglabels=["Personal Care","Household Care","Work/Education","Consumerism","Eating/Drinking","Social/Recreation","Travel"]
	for i in range(number_of_lines):
		legart += [mpatch.Rectangle((0,0),2,2,fc=maincolors[i],hatch=hatchset[i])]

	framegroup = frame.groupby(labelcolumn);
	labelscount = Counter(frame[frame["TUACTIVITY_N"] == 1][labelcolumn]);
	#jg is for joingroup
	c = 1
	for jg in framegroup:
		# if jg[0] not in [24,37,53,76]:
		if jg[0] not in [45,33,79,154]:
			continue;
		
		if(labelscount[jg[0]] < cutoff):
			continue;
		
		daycount = np.zeros(2)
		data = np.zeros([actcount,288*2])
		sum = np.zeros(288*2)
		dayset = {1}
		
		# #print(jg[1])
		# subsetjg = jg[1][jg[1]['TUACTIVITY_N'] == 1]
		# mdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[1]
		# fdem = subsetjg.groupby(["TESEX","TEAGE"]).TESEX.count()[2]
		# #print(mdem,fdem);
		# if(type(mdem) == pd.core.series.Series): mdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='b')
		# else: print("bad mdem:",mdem,type(mdem))
		# if(type(fdem) == pd.core.series.Series): fdem.reindex(np.arange(0,101)).fillna(int(0)).plot(color='r')
		# else: print("bad fdem:",fdem,type(fdem))

		# plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		# F = plt.gcf();
		# F.set_size_inches(4,3)
		# F.set_dpi(300);
		# F.savefig(prefix + "demo-" +str(jg[0]) + ".png");
		# plt.clf();
		
		
		for ind, row in jg[1].iterrows():
			
			fullcode = tri[int(row['TUTIER1CODE'])]
			# fullcode = tri[int(row['TRCODE'])]
			
			# day =  int(row['TUDIARYDAY']) - 1;
			# day =  (0 if int(row['TUDIARYDAY']) in weekdayset else 1);
			day = 0
			dayset.add(day);
			daycount[day] += 1;
			
			stop = np.floor(row['TUCUMDUR24']/5.0);
			start = stop - np.floor(row['TUACTDUR24']/5.0)
			startind = int(day * 288 + start)
			stopind = int(day * 288 + stop)
		
			data[fullcode,startind:stopind] += 1;
		
			sum[startind:stopind] += 1;

		#print("normalizing...")
		for i in range(len(data)):
			data[i] /= sum;

		# if(savemats):
		# 	ds = h5out.create_dataset('/label-'+str(jg[0]),data=data,fillvalue=0.,compression='gzip',compression_opts=9);
		# 	ds.attrs['label']=jg[0];
		# 	ds.attrs['prefix']=prefix;
		



		
		print("plotting label "+ str(jg[0]) + ", c:"+str(labelscount[jg[0]]))
		# x = np.arange(1.0,(8.0+(4.0/24.0)),(8.-1.0)/(288*7+48))
		x = np.arange(0.0,2.0,1.0/288.0)
		
		additive = np.zeros(288*2)
		
		# ax = plt.subplot(111)
		ax = plt.subplot(2,2,c)
		box = ax.get_position()
		# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.set_xlim([0.0,0.99])
		ax.set_ylim([0.0,1.0])
		ax.set_xticks([0.0,0.25,0.5,0.75,0.99])
		if(c in [3,4]):
			ax.set_xticklabels(["4:00","10:00","16:00","22:00","4:00"])
		else:
			ax.set_xticklabels([])
		if(c in [2,4]):
			ax.set_yticklabels([])
		lpha = 1.0
		datasum = np.cumsum(data, axis=0);
		ax.fill_between(x,0,datasum[0],edgecolor='k',facecolor=colors[0],hatch=hatchset[0],linewidth=0.5,alpha=lpha)
		
		for i in range(actcount-1):
			ax.fill_between(x,datasum[i],datasum[i+1],edgecolor='k',facecolor=colors[i+1],hatch=hatchset[i+1],linewidth=0.5,alpha=lpha)

		# 
		# plt.title( "Label " +str(jg[0])+", count: "+str(labelscount[jg[0]]) )
		# F = plt.gcf();
		# F.set_size_inches(12,8)
		# F.set_dpi(300);
		# F.savefig(prefix +"label-" +str(jg[0]) + ".png");
		# plt.clf();
		c += 1

	F = plt.gcf();
	F.set_size_inches(8. ,4.)
	F.set_dpi(100);
	plt.tight_layout()
	plt.legend(legart[::-1],leglabels[::-1],ncol=4,loc='lower center', bbox_to_anchor=(-0.2, -.6))
	plt.show();




print("processing...")

# for g,frame in acttable.groupby('daytypelabelreduce'):
demoActPlot(acttable,"daytypelabelreduce")
