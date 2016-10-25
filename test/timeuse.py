import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
import numpy as np;

sys.path.append("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/test")
import bettersankey as bsk;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')


# jointable.groupby(['TUDIARYDAY','TUTIER1CODE','TUTIER2CODE']).count().TUCASEID.plot(kind='bar')
# plt.show()
# 
# exit()
#for each activity record find the 

daycount = np.zeros(7)
data = np.zeros([18,288*7])
sum = np.zeros(288*7)
dayset = {1}



print("processing...")
for ind, row in jointable.iterrows():
	
	codetier1 = int(row['TUTIER1CODE']) - 1
	if(codetier1 > (18 - 1)):
# 	if(codetier1 !=(17)):
		continue;
# 	
# 	codetier2 = int(row['TUTIER2CODE']) - 1;
# 	if(codetier2 == 98): codetier2 = 17 - 1;
# 	if(codetier2 > (18 - 1)):
# 		continue;
# 	 
	day = int(row['TUDIARYDAY']) - 1;
	dayset.add(day);
	daycount[day] += 1;
	#t = time.strptime(row['TUSTARTTIM'],'%H:%M:%S')
	#start = np.floor((t.tm_hour*60+t.tm_min)/5)
	#t = time.strptime(row['TUSTOPTIME'],'%H:%M:%S')
	#stop = np.floor((t.tm_hour*60+t.tm_min)/5)
	
	
	stop = np.floor(row['TUCUMDUR24']/5.0);
	start = stop - np.floor(row['TUACTDUR24']/5.0)
# 	if(start == stop): #implicitly, we don't expect someone to do the same thing for 24 hours straight unless they are messed up
# 		stop = start + 1;
	startind = int(day * 288 + start)
	stopind = int(day * 288 + stop)
	
# 	if(stop < start):
# 		stopind += 288;
# 		if(stopind > (day*288+288+48)):
# 			stopind = day*288+288+48;


	data[codetier1,startind:stopind] += 1;
# 	data[codetier2,startind:stopind] += 1;



	sum[startind:stopind] += 1;
	#else:
		
# 		data[codetier1,startind:(day*288+288)] += 1;
# 		data[codetier1,(day*288):stopind] += 1;
		
print("normalizing...")
for i in range(len(data)):
	data[i] /= sum;


print("set of day vals: ",dayset);

print("plotting...")
# x = np.arange(1.0,(8.0+(4.0/24.0)),(8.-1.0)/(288*7+48))
x = np.arange(0.0,7.0,1.0/288.0)


start = 0.0
stop = 1.0
number_of_lines= 18
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ cm.prism(x) for x in cm_subsection ]

additive = np.zeros(288*7)

ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_ylim([0.0,1.0])

legart = []
leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling"]

datasum = np.cumsum(data, axis=0);
ax.fill_between(x,0,datasum[0],facecolor=colors[0],alpha=0.7,linewidth=0.1)
legart += [mpatch.Rectangle((0,0),1,1,fc=colors[0])]

for i in range(len(colors)-1):
	ax.fill_between(x,datasum[i],datasum[i+1],facecolor=colors[i+1],alpha=0.7,linewidth=0.1)
	legart += [mpatch.Rectangle((0,0),1,1,fc=colors[i+1])]
# 	additive += data[i];
# 	plt.plot(x,datasum[i],linewidth=2.0,color=colors[i]);
# 	plt.plot(x,data[i],color=c,linewidth=4.0);


plt.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	
plt.show();
	
