import pandas as pd;
import numpy as np;
import sqlite3;
import collections;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

#load indvs
con = sqlite3.connect(datapath + "indvs.sq3");
indvs = pd.read_sql("select * from indvs", con);
con.close();
#load locations
con = sqlite3.connect(datapath + "blockaddr.sq3");
blockaddr = pd.read_sql("select * from blockaddr", con);
con.close();
#load LEHD tables
odmat = pd.read_csv(datapath + "lehd/ut_od_main_JT00_2014.csv")
#rac = pd.read_csv(datapath + "lehd/ut_rac_S000_JT00_2014.csv")
wac = pd.read_csv(datapath + "lehd/ut_wac_S000_JT00_2014.csv")

# print(indvs)
# print(blockaddr)
# print(odmat)
# print(rac)
# print(wac)

out = pd.DataFrame();

out['id'] = indvs['id'];
out['block'] = indvs['block'];
out['age'] = indvs['age'].apply(lambda x: 0 if x >= 15 else 1);
out['empblock'] = -1
out['empx'] = 0.0;
out['empy'] = 0.0;
out['empcode'] = -1; 

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
		
		max = len(df) if len(odg) > len(df) else len(odg)
		out.loc[df.index[:max],['empblock']] = wlist[:max];
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



