import pandas as pd;
import numpy as np;
import sqlite3;
import collections;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

print("loading...")
# #load indvs
con = sqlite3.connect(datapath + "indvs2.sq3");
indvs = pd.read_sql("select * from indvs", con);
con.close();
# #load locations
con = sqlite3.connect(datapath + "blockaddr2.sq3");
blockaddr = pd.read_sql("select * from blockaddr", con);
con.close();
#load LEHD tables
odmat = pd.read_csv(datapath + "lehd/ut_od_main_JT00_2014.csv")
#rac = pd.read_csv(datapath + "lehd/ut_rac_S000_JT00_2014.csv")
wac = pd.read_csv(datapath + "lehd/ut_wac_S000_JT00_2014.csv")

youngtab = pd.read_csv(datapath + "employ/ACS_15_5YR_B23022_with_ann.csv",skiprows=[1])
oldtab = pd.read_csv(datapath + "employ/ACS_15_5YR_B23026_with_ann.csv",skiprows=[1])
agetab = pd.read_csv(datapath + "employ/ACS_15_5YR_B23001_with_ann.csv",skiprows=[1])

# print(indvs)
# print(blockaddr)
# print(odmat)
# print(rac)
# print(wac)

indvs = indvs.iloc[0:1000];


print("building brackets...")

#hours/weeks scheduling
agebrackets = [16,20,22,25,30,35,45,55,60,62,65,70,75,101];

ageindices = np.array([3,10,17,24,31,38,45,52,59,66,73,79,85])
agediff = list(np.diff(ageindices) - 2);
agediff += [agediff[len(agediff)-1]]
#print(agediff)
gendershift = [0,92-3]

probbracket = []
for i in gendershift:
	temp = [];
	for j,k in zip(ageindices,agediff):
		total = agetab['HD01_VD'+str(i+j).zfill(2)][0]
		nonemploy = agetab['HD01_VD'+str(i+j+k).zfill(2)][0]
		nonlabor = agetab['HD01_VD'+str(i+j+k+1).zfill(2)][0]
		employ = total - nonemploy - nonlabor
		temp += [employ / total];
	probbracket += [temp]



#it's not terribly reasonable to have more than 70 hours
#really the distribution is more complicated than simple breaks
#I will probably need special distributions
hourbrackets = [(1,15),(15,35),(35,60)];
weekbrackets = [(50,52),(48,50),(40,48),(27,40),(14,27),(1,14)];
hourweekbrackets = []
for i in hourbrackets:
	for j in weekbrackets:
		hourweekbrackets += [(i,j)]

def hourbracketbuild(table):
	totalindices = [4,29];
	hourindices = [19,12,5];
	weekindices = range(1,7);
	gendershift = [0,30-5];
	bracket = []
	for i,j in zip(gendershift,totalindices):
		temp = [];
		total = table['HD01_VD'+str(j).zfill(2)][0]
		for k in hourindices:
			for m in weekindices:
				count = table['HD01_VD'+str(i+k+m).zfill(2)][0]
				#print(count, str(i+k+m).zfill(2));
				temp += [count / total];
		bracket += [temp]
	return bracket;

youngbracket = hourbracketbuild(youngtab)
oldbracket = hourbracketbuild(oldtab);


#in the future I could interpolate with a lower bound cutoff 
#instead of using the bracket system, but this is inconsequential at the moment
def calcprob(x,agebrackets,prob):
	
	#calc employ prob
	if(x.age < 16 and x.age >= 14): return 0.01;
	if(x.age < 14): return 0.0;

	for i in range(len(agebrackets) - 1):
		if(x.age < agebrackets[i+1] and x.age >= agebrackets[i]):
			p = prob[int(x.gender)][i];
			p = 1.0 if p > np.random.random() else p;
			return p;
	return 0.0;



print("building outframe...")

out = pd.DataFrame();

out['id'] = indvs['id'];
out['block'] = indvs['block'];
out['gender'] = indvs['gender'];
out['age'] = indvs['age']
out['empblock'] = -1
out['empx'] = 0.0;
out['empy'] = 0.0;
out['empcode'] = -1; 
out['hours']=0 #determines daily schedule
out['weeks']=0 #determines weekly work prob
out['shift']=0 #determines normal shift, or alternative shift
out['probemploy'] = out.apply(calcprob, args=(agebrackets,probbracket), axis=1);



print("assigning blocks...")
print(out);

gout = out.groupby(['block'])

odout = odmat.groupby(['h_geocode']);
#DRAFT VERSION
c = 0;
for g,df in gout:
	#print(g)
	#print(df)
	df = df.sort_values('probemploy',axis=0,ascending=False)
	
	dfcount = np.sum(df['probemploy'] > 0.0);
	#print(df)
	if(g in odout.indices):
		
		odg = odout.get_group(g)[['w_geocode','S000']];
		#print(odg)
		
		counter = collections.Counter();
		key = list(odg['w_geocode'])
		count = list(odg['S000']);
		k = zip(key,count);
		for i in k:
			counter[int(i[0])] = int(i[1]);
		wlist = list(counter.elements())
		
		np.random.shuffle(wlist);
		
		#maxnum = len(df) if len(odg) > len(df) else len(odg)
		maxnum = dfcount if len(odg) > dfcount else len(odg);
		print('maxnum: ',maxnum);
		out.loc[df.index[:maxnum],['empblock']] = wlist[:maxnum];
		#print(out.iloc[df.index])

# 	c+=1;
# 	if(c == 3): break;

print("assigning employment address...")
	
gout = out.groupby(['empblock'])
bgroup = blockaddr.groupby(['block'])

def pickxycode(x,bg,wg):
	if(x['empblock'] != -1):
		sz = len(bg)
		pick = np.random.randint(0,sz);
		x['empx'] = bg.iloc[pick]['addrx']
		x['empy'] = bg.iloc[pick]['addry']
		x['empcode'] = np.random.choice(20,p=wg);
	return x;


	
cols = []
for i in range(1,21):
	cols += ["CNS"+str(i).zfill(2)]


c = 0;
for g,df in gout:
	if(g in bgroup.indices):
		bg = bgroup.get_group(g);
		wg = np.array(wac.loc[(wac['w_geocode'] == g), cols].iloc[0])
		wg = wg/np.sum(wg);

		out.loc[df.index] = df.apply(pickxycode,axis=1,args=(bg,wg,));

		#print(out.loc[df.index])

# 	c+=1;
# 	if(c == 3): break;


print("writing...");

out = out.drop('block',axis=1);
out = out.drop('age',axis=1);
out = out.drop('gender',axis=1);
print(out);

exit()
con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/employ-"+time.strftime("%Y-%m-%d_%H-%M-%S")+".sq3");
out.to_sql('employ',con);
con.close();



