import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import matplotlib.patches as mpatch;
from matplotlib.collections import PatchCollection;
import numpy as np;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
outpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/test/"

simseqs = pd.read_csv(datapath + "simseqs.csv")
# imgpath = outpath + "singleact-" + time.strftime("%Y-%m-%d_%H-%M-%S")
# os.mkdir(imgpath)

print("loading...")

acttable = pd.read_csv(datapath + "timeuse/atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "timeuse/atusresp_2015/atusresp_2015.dat")
acttable = pd.merge(acttable,infotable[['TUCASEID','TUDIARYDAY']],on='TUCASEID')

labels = pd.read_csv(datapath + "final-label-classifier/labels.csv")

acttable = pd.merge(acttable,labels[['daytypelabelreduce','TUCASEID']],on='TUCASEID')


acttable['start'] = (acttable['TUCUMDUR24']-acttable['TUACTDUR24'])/60.0
acttable['length'] = acttable['TUACTDUR24']/60.0

actmapping = np.sort(list(set(acttable['TRCODE'])))

ati = { tr:i for i,tr in enumerate(actmapping) }
ita = { i:tr for i,tr in enumerate(actmapping) }
# print([(i,k) for i,k in enumerate(actmapping)]);

locmapping = np.sort(list(set(acttable['TEWHERE'])))
wti = { tr:i for i,tr in enumerate(locmapping) }
itw = { i:tr for i,tr in enumerate(locmapping) }
# print([(i,k) for i,k in enumerate(locmapping)]);

acttable['mapped'] = acttable['TRCODE'].apply(lambda x: ati[x]);


#set up coloration
start = 0.0
stop = 1.0
number_of_lines=19
cm_subsection = np.linspace(start, stop, number_of_lines)
# np.random.shuffle(cm_subsection)
maincolors = [ cm.prism(x) for x in cm_subsection ]
colors = []
for i in actmapping:
	tier1 = i // 10000;
	if (tier1 == 50): tier1 = 19;
	tier1 = tier1 - 1;
	
	scol = list(maincolors[tier1])
# 	scol[2] = scol[2] + (tier2 * 0.1)
# 	if(scol[2] > 1.0): scol[2] = 1.0;
# 	scol[3] = scol[3] + (tier2 * 0.1)
# 	if(scol[3] > 1.0): scol[3] = 1.0;
	colors += [list(scol)];


acttable["color"] = acttable['mapped'].apply(lambda x: colors[x])
simseqs["color"] = simseqs["actind"].apply(lambda x: colors[x])

# print(acttable["color"])

#set up labeling
legart = []
leglabels=["Personal Care","HH activities","HH Member Care","NonHH care","work","Edu", "Consumerism", "Pro Services", "HH services", "Gov services", "Eating/Drinking","SocialRelaxLeisure","Sports/Excercise","Religious","Volunteering", "Telephone","Null","Traveling", "Other"]
for i in range(number_of_lines):
	legart += [mpatch.Rectangle((0,0),1,1,fc=maincolors[i])]

weekdayset = [2,3,4,5,6]


def altplotseq(frame,label):
	fig, ax = plt.subplots()

	cneg = -1
	cpos = 1;




	for i,gr in enumerate(frame.groupby("agentnum")):
		g,df = gr;
		# df = df.sort_values(["TUACTIVITY_N"])
		patches = [];


		# day =  int(df['TUDIARYDAY'][0]) - 1;
		# day =  (1 if df['TUDIARYDAY'].iloc[0] in weekdayset else 0);
		day = 1

		x = df['start'].values;
		y = np.zeros_like(x)

		if(day): 
			y[:] = cneg
			cneg -=1
		else:
			y[:] = cpos
			cpos += 1

		c = df['color'].values;
		w = df['length'].values;
		h = np.ones_like(w)
		
		for xi,yi,wi,hi,ci in zip(x,y,w,h,c):
			patches.append(mpatch.Rectangle((xi,yi),wi,hi,facecolor=ci,edgecolor='black',linewidth=0.5))
		
		p = PatchCollection(patches,match_original=True)

		ax.add_collection(p)

		# cn+=1;
		# if(cn > 500): break;
		

	# fig.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	#print(cn);

	ax.set_xlim((0.,1440.))
	ax.set_xticks((0,360.,720.,1080,1440.))
	ax.set_xticklabels(("4:00", "10:00", "16:00", "22:00","4:00"))
	ax.set_yticks(());
	# ax.set_ylim((cneg,cpos))
	ax.set_ylim(-24.0,0.0)
	ax.set_aspect(60.0)

	ax.set_title(label)
	# plt.show()


def plotseq(frame,label):
	fig, ax = plt.subplots()

	cneg = -1
	cpos = 1;




	for i,gr in enumerate(frame.groupby("TUCASEID")):
		g,df = gr;
		df = df.sort_values(["TUACTIVITY_N"])
		patches = [];


		# day =  int(df['TUDIARYDAY'][0]) - 1;
		day =  (1 if df['TUDIARYDAY'].iloc[0] in weekdayset else 0);

		x = df['start'].values;
		y = np.zeros_like(x)

		if(day): 
			y[:] = cneg
			cneg -=1
		else:
			y[:] = cpos
			cpos += 1

		c = df['color'].values;
		w = df['length'].values;
		h = np.ones_like(w)
		
		for xi,yi,wi,hi,ci in zip(x,y,w,h,c):
			patches.append(mpatch.Rectangle((xi,yi),wi,hi,facecolor=ci,edgecolor='black',linewidth=0.5))
		
		p = PatchCollection(patches,match_original=True)

		ax.add_collection(p)

		# cn+=1;
		# if(cn > 500): break;
		

	# fig.legend(legart,leglabels,loc='center left', bbox_to_anchor=(1, 0.5))
	#print(cn);

	ax.set_xlim((0.,24.))
	ax.set_xticks((0,6,12,18,24))
	ax.set_xticklabels(("4:00", "10:00", "16:00", "22:00","4:00"))
	ax.set_yticks(());
	# ax.set_ylim((cneg,cpos))
	ax.set_ylim(-24.0,0.0)
	ax.set_aspect(1.0)

	ax.set_title(label)

	# plt.show()
	# plt.show()

for g,frame in acttable.groupby('daytypelabelreduce'):
	if(g == 14):
		print(g,len(frame))
		plotseq(frame,str(g)+",real");

for g,frame in simseqs.groupby('daytype'):
	if(g == 14):
		print(g,len(frame))
		altplotseq(frame,str(g)+",real");

plt.show()