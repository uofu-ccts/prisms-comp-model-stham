import pandas as pd;
import time;
import matplotlib.pyplot as plt;
import numpy as np;
import sklearn.ensemble
import sklearn.tree
from sklearn.externals import joblib;
import sqlite3;

from collections import Counter;

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
#id,schoolprob,schoolevel
con.close();

mergetab = pd.merge(indvs,employ,on='id')
mergetab = pd.merge(mergetab,school,on='id')
mergetab = mergetab.drop(['index','index_x','index_y'],axis=1)
print(mergetab.columns);

#clf = joblib.load(datapath + "NEEDPATH")

goodcols = ['TEAGE', 'TEHRUSL1', 'TELFS', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESEX', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRHHCHILD', 'TRSPPRES', 'TUDIS2', 'TUELNUM', 'TUSPUSFT']

mux = pd.DataFrame();

def age(x):
	if x >= 80 and x < 85: return 80;
	if x >= 85: return 85;
	return x;

mux['TEAGE'] = indvs['age'].apply
