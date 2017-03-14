import pandas as pd;
import numpy as np;
import sqlite3;
import collections;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

# #load indvs
# con = sqlite3.connect(datapath + "indvs.sq3");
# indvs = pd.read_sql("select * from indvs", con);
# con.close();
# #load locations
# con = sqlite3.connect(datapath + "blockaddr.sq3");
# blockaddr = pd.read_sql("select * from blockaddr", con);
# con.close();
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

#hours/weeks scheduling
agebrackets = [16,20,22,25,30,35,45,55,60,62,65,70,75,101];

ageindices = np.array([3,10,17,24,31,38,45,52,59,66,73,79,85])
agediff = list(np.diff(ageindices) - 2);
agediff += [agediff[len(agediff)-1]]
print(agediff)
ageshift = [0,92-3]

probbracket = []
for i in ageshift:
	temp = [];
	for j,k in zip(ageindices,agediff):
		print(j,k)
		total = agetab['HD01_VD'+str(i+j).zfill(2)][0]
		nonemploy = agetab['HD01_VD'+str(i+j+k).zfill(2)][0]
		nonlabor = agetab['HD01_VD'+str(i+j+k+1).zfill(2)][0]
		employ = total - nonemploy - nonlabor
		temp += [employ / total];
	probbracket += [temp]


print(agebrackets);
print(probbracket);

hourbrackets = [0,1,15,35];
weekbrackets = [0,1,14,27,40,48,50];



exit()


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
out['probemploy'] = out.apply(calcprob,args=(ageprobtables), axis=1);


gout = out.groupby(['block'])

odout = odmat.groupby(['h_geocode']);
#DRAFT VERSION
c = 0;
for g,df in gout:
	#print(g)
	#print(df)
	df = df.sort(['age'])
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
		
		maxnum = len(df) if len(odg) > len(df) else len(odg)
		out.loc[df.index[:maxnum],['empblock']] = wlist[:maxnum];
		#print(out.iloc[df.index])

# 	c+=1;
# 	if(c == 3): break;





	
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


out = out.drop('block',axis=1);
out = out.drop('age',axis=1);

con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/employ.sq3");
out.to_sql('employ',con);
con.close();



