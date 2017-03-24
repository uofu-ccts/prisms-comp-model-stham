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


con = sqlite3.connect(datapath + "indvs2.sq3");
indvs = pd.read_sql("select * from indvs limit 1000", con);
#age,g1,g2,gender,household,householder,group,mobile,block,addrx,addry,addrn,city,id,spouse
con.close();
con = sqlite3.connect(datapath + "employ2.sq3");
employ = pd.read_sql("select * from employ limit 1000", con);
#id,empblock,empx,empy,empcode,emphours,empweeks,empshift,probemploy
con.close();
con = sqlite3.connect(datapath + "school2.sq3");
school = pd.read_sql("select * from school limit 1000", con);
#id,schoolprob,schoollevel
con.close();

mergetab = pd.merge(indvs,employ,on='id')
mergetab = pd.merge(mergetab,school,on='id')
mergetab = mergetab.drop(['index','index_x','index_y'],axis=1)
print(mergetab.columns);
print(mergetab);

#clf = joblib.load(datapath + "NEEDPATH")


print("transforming dataset to classfier...")

goodcols = ['TEAGE', 'TEHRUSL1', 'TELFS', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESEX', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRHHCHILD', 'TRSPPRES', 'TUDIS2', 'TUELNUM', 'TUSPUSFT']

mux = pd.DataFrame();

mux['id'] = mergetab['id']

def ageClip(x):
	if x >= 80 and x < 85: return 80;
	if x >= 85: return 85;
	return x;

mux['TEAGE'] = mergetab['age'].apply(ageClip);

mux['TEHRUSL1'] = mergetab['emphours'];


def telfsClass(x):

	if x.emphours > 0: return 1;
	return 5;

mux['TELFS'] = mergetab.apply(telfsClass,axis=1);


mux['TESCHENR'] = mergetab['schoollevel'].apply(lambda x: 1 if x > -1 else 2);

def teschftClass(x):
	#full/part time is determined by 
	#employment and school level
	#the probabilties are imputes from ATUS 2015 
	#using counts of TESCHLVL, TELFS, and TESCHFT
	if x.schoollevel == 3: #highschool
		if x.emphours > 1:
			return 1 if (0.87179) > np.random.random() else 2
		else:
			return 1 if (0.97772) > np.random.random() else 2
	elif x.schoollevel == 4:
		if x.emphours > 1: #college
			return 1 if (0.52647) > np.random.random() else 2
		else:
			return 1 if (0.72000) > np.random.random() else 2
	return -1;

mux['TESCHFT'] = mergetab.apply(teschftClass,axis=1);

def teschlvlClass(x):
	if x == 3: return 1;
	if x == 4: return 2;
	return -1;

mux['TESCHLVL'] = mergetab['schoollevel'].apply(teschlvlClass);

mux['TESEX'] = mergetab['gender'].apply(lambda x: x+1);



householdg = mergetab.groupby('household')

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

mux['TRCHILDNUM'] = 0;
mux['TUELNUM'] = -1;
mux['TRSPPRES'] = 3
mux['TESPEMPNOT'] = -1
mux['TESPUHRS'] = -1
mux['TUSPUSFT'] = -1

for gr,df in householdg:

	child = 0; eld = 0
	for ind,i in df.iterrows():
		
		if(i.age >= 80): eld += 1;
		if(i.age < 18): child += 1;

	#0 elders does not exist
	if(eld == 0): eld = -1;

	sp = df[df['spouse']>-1];
	#implicitely, there are only two spouses in a household because of
	#how households are constructed. 
	if(len(sp) > 0):
		a = sp.iloc[0];
		b = sp.iloc[1];
		#print(a.id,b.id)
		trsppres = trsppresClass(a.age,b.age);
		mux.loc[a.id,['TRSPPRES']] = trsppres;
		mux.loc[b.id,['TRSPPRES']] = trsppres;
		mux.loc[a.id,['TESPEMPNOT']] = 1 if b.emphours > 0 else 2;
		mux.loc[b.id,['TESPEMPNOT']] = 1 if a.emphours > 0 else 2;
		mux.loc[a.id,['TESPUHRS']] = b.emphours;
		mux.loc[b.id,['TESPUHRS']] = a.emphours;
		mux.loc[a.id,['TUSPUSFT']] = 1 if b.emphours > 35 else 2;
		mux.loc[b.id,['TUSPUSFT']] = 1 if a.emphours > 35 else 2;


	mux.loc[df.index,['TRCHILDNUM']] = child;
	mux.loc[df.index,['TUELNUM']] = eld;

#this value has a low low low feature importance because it doesn't say very much
#mux['TUSPUSFT'] = mux['TESPUHRS'].apply(lambda x: 1 if x > 35 else -1)


#mux['hh'] = mergetab['household']


def trdpftptClass(x):
	if x >= 35 : return 1;
	elif x > 0 : return 2;
	return -1;

mux['TRDPFTPT'] = mergetab['emphours'].apply(trdpftptClass);

mux['TRHHCHILD'] = mux['TRCHILDNUM'].apply(lambda x: 1 if x > 0 else 2);

mux['TUDIS2'] = mergetab['mobile'].apply(lambda x: (1 if 0.5 > np.random.random() else 2) if x < 1 else -1);

print(mux)
print(mux.columns)