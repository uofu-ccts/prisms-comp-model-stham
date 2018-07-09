import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import numpy as np;
import sklearn.ensemble
import sklearn.tree
from sklearn.externals import joblib;
import sqlite3;

from collections import Counter;

pd.set_option('display.max_columns', 1000)
#pd.set_option('display.max_rows', 1000)

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

print("loading...")

limiter = ""

con = sqlite3.connect(datapath + "indvs2.sq3");
indvs = pd.read_sql("select * from indvs" + limiter, con);
#age,g1,g2,gender,household,householder,group,mobile,block,addrx,addry,addrn,city,id,spouse
con.close();
con = sqlite3.connect(datapath + "employ2.sq3");
employ = pd.read_sql("select * from employ" + limiter, con);
#id,empblock,empx,empy,empcode,emphours,empweeks,empshift,probemploy
con.close();
con = sqlite3.connect(datapath + "school2.sq3");
school = pd.read_sql("select * from school" + limiter, con);
#id,schoolprob,schoollevel
con.close();

mergetab = pd.merge(indvs,employ,on='id')
mergetab = pd.merge(mergetab,school,on='id')
mergetab = mergetab.drop(['index','index_x','index_y'],axis=1)
# print(mergetab.columns);
# print(mergetab);

alttab = mergetab[mergetab['spouse'] > -1]

alttab = alttab[['spouse','emphours','age']].set_index('spouse')


#print(mergetab.columns,alttab.columns);

mergetab = mergetab.join(alttab,how='left',rsuffix='sp')

#print(mergetab[['spouse','age','agesp','emphours','emphourssp']])


clfpath = datapath + "final-label-classifier/clfsave-casetype.pkl"
#print(clfpath)
clf = joblib.load(clfpath)


print("transforming dataset to classfier...")

goodcols = ['TEAGE', 'TEHRUSL1', 'TELFS', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESEX', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRHHCHILD', 'TRSPPRES', 'TUDIS2', 'TUELNUM', 'TUSPUSFT']

mux = pd.DataFrame();

mux['id'] = mergetab['id']

def ageClip(x):
	if x >= 80 and x < 85: return 80;
	if x >= 85: return 85;
	return x;

mux['TEAGE'] = mergetab['age'].apply(ageClip);


#this is a bug: tehrus can be -1, which reallyu throws a curve on a couple branches
#in the 2015 ATUS dataset, only 9 respondents gave zero hours for work
#therefore we can safely infer that anyone with 0 emphours is correctly classed
#as TEHRUSL1 = -1, which matches edited universe notion of TELFS = 1 or 2
mux['TEHRUSL1'] = mergetab['emphours'].apply(lambda x: -1 if x < 1 else x);


def telfsClass(x):

	if x.emphours > 0: return 1;
	#the only real class that appears important besides 1 and 5 is 4 and 2; no easy way to infer
	return 5;

mux['TELFS'] = mergetab.apply(telfsClass,axis=1);

#also wrong; teschenr is an edited universe with resp 15-49
#this can probably be adjusted to 5-49 for the sake of accuracy

def teschenrClass(x):
	if(x.age <= 49 and x.age >= 5):
		return -1;
	else: 
		if(x.schoollevel > -1): 
			return 1;
		else: 
			return 2;

mux['TESCHENR'] = mergetab.apply(teschenrClass,axis=1);

def teschftClass(x):
	#full/part time is determined by 
	#employment and school level
	#the probabilties are imputes from ATUS 2015 
	#using counts of TESCHLVL, TELFS, and TESCHFT
	if x.schoollevel == 3: #highschool
		if x.emphours >= 1:
			return 1 if (0.87179) > np.random.random() else 2
		else:
			return 1 if (0.97772) > np.random.random() else 2
	elif x.schoollevel == 4:
		if x.emphours >= 1: #college
			return 1 if (0.52647) > np.random.random() else 2
		else:
			return 1 if (0.72000) > np.random.random() else 2
	return -1;

mux['TESCHFT'] = mergetab.apply(teschftClass,axis=1);

def teschlvlClass(x):
	if x == 1: return 1;
	if x == 2: return 1;
	if x == 3: return 1;
	if x == 4: return 2;
	return -1;

mux['TESCHLVL'] = mergetab['schoollevel'].apply(teschlvlClass);

mux['TESEX'] = mergetab['gender'].apply(lambda x: x+1);





def trsppresClass(a,b):
	#in order to deal with class 2 (unmarried partner present)
	#we say that a fraction of "spouses" are actually unmarried
	#this is probably a miscalculation since households doesn't 
	#use any census tables that address marriage status, 
	#and because the census seems to treat this class as non-material 
	#for household classification
	#a linear function of y = -0.055x+1.7 describes the probability
	#that a married couple is actually unmarried, up to age 30
	#above age 30 a flat probability of 0.05 is assigned. 
	#below age 20 a probability of 0.5 is used.
	#this only applies to individuals who have a spouse assigned
	avgage = (a+b)/2
	if(avgage < 20): return 2 if 0.5 > np.random.random() else 1;
	elif(avgage <30): return 2 if (-0.055*avgage+1.7) > np.random.random() else 1;
	else: return 2 if 0.05 > np.random.random() else 1;
	return 3;

#mux['TRSPPRES'] 

# mux['TRCHILDNUM'] = 0;
# mux['TUELNUM'] = -1;
# mux['TRSPPRES'] = 3
# mux['TESPEMPNOT'] = -1
# mux['TESPUHRS'] = -1
# mux['TUSPUSFT'] = -1

print(" spousemerging....")

householdg = mergetab.groupby('household')

def spouseapply(x):
	#default values for non-spousal stuff
	if(x.spouse == -1):
		return 3,-1,-1,-1
	sppres = trsppresClass(x.age,x.agesp)
	
	empnot = (1 if x.emphourssp > 0 else 2) if sppres < 3 else -1;
	spuhrs = -1 if x.emphourssp < 0 else x.emphourssp
	spusft = 1 if x.emphourssp > 35 else 2;

	return sppres,empnot,spuhrs,spusft;

mux['TRSPPRES'],mux['TESPEMPNOT'],mux['TESPUHRS'],mux['TUSPUSFT'] = zip(*mergetab.apply(spouseapply,axis=1))

print(" hhcounts...")

hhcounts = {}
for gr,df in householdg:

	child = 0; eld = 0
	for ind,i in df.iterrows():
		
		if(i.age >= 80): eld += 1;
		if(i.age < 18): child += 1;

	#0 elders does not exist
	if(eld == 0): eld = -1;
	hhcounts[gr] = (child,eld);

mux['TRCHILDNUM'] = mergetab['household'].apply(lambda x: hhcounts[x][0]);
mux['TUELNUM'] = mergetab['household'].apply(lambda x: hhcounts[x][1] if hhcounts[x][1] > 0 else -1);


# 	mux.loc[df.index,['TRCHILDNUM']] = child;
# 	mux.loc[df.index,['TUELNUM']] = eld;


#mux['hh'] = mergetab['household']

print(" remaining bits...")

def trdpftptClass(x):
	if x >= 35 : return 1;
	elif x > 0 : return 2;
	return -1;

mux['TRDPFTPT'] = mergetab['emphours'].apply(trdpftptClass);

mux['TRHHCHILD'] = mux['TRCHILDNUM'].apply(lambda x: 1 if x > 0 else 2);

mux['TUDIS2'] = mergetab['mobile'].apply(lambda x: (1 if 0.5 > np.random.random() else 2) if x < 1 else -1);

# print(mux)
# print(mux.columns)


print("classifying...")

out = pd.DataFrame();
out['id'] = mergetab['id']

vals = mux[goodcols].values;


labels = [];
for i in range(0,len(vals),1000):
	last = i + 1000;
	if(last > len(vals)): last = len(vals);
	labels += [clf.predict(vals[i:last])]
out['casetype'] = np.concatenate(labels);

#print(out);

print("writing...")

out.to_csv(datapath + "indvlabels.csv")
