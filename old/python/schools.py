import pandas as pd;
import numpy as np;
import sqlite3;
import collections;
import time;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
print("loading...")
# #load indvs
con = sqlite3.connect(datapath + "indvs2.sq3");
indvs = pd.read_sql("select * from indvs", con);
con.close();

schooltab = pd.read_csv(datapath + "schools/ACS_15_5YR_B14003_with_ann.csv",skiprows=[1])
#we don't need the second table for college enrollment;
#since birthdate isn't taken into account, we will just roll 17 year olds into a cohort
#that will attend HS and all 18 years olds in school will be in college of some kind
#collegetab = pd.read_csv(datapath + "schools/ACS_15_5YR_B14004_with_ann.csv",skiprows=[1])

print("building brackets...")

agebracket = [3,5,10,15,18,20,25,35,101]
ageindices = list(range(4,12));
gendershift = [0,32-4]
#indices = [0,9,18];

schoolprob = []
for i in gendershift:
	temp = [];
	for j in ageindices:
		public = schooltab['HD01_VD'+str(i+j).zfill(2)][0]
		private = schooltab['HD01_VD'+str(i+j+9).zfill(2)][0]
		non = schooltab['HD01_VD'+str(i+j+18).zfill(2)][0]
		total = non+public+private
		temp += [ (public+private)/ total ]
	schoolprob += [temp];

#print(schoolprob);

def calcschoolprob(x,agebracket,prob):

	for i in range(len(agebracket) - 1):
		if(x.age < agebracket[i+1] and x.age >= agebracket[i]):
			p = prob[int(x.gender)][i];
			#this is not the place to calculate the probability because
			#we might want the values later
			#p = 1.0 if p > np.random.random() else p;
			return p;
	return 0.0;

gradebracket = [0,5,12,14,18,101]
def assignschools(x,bracket):
	#-1 none, 0 pre, 1 elem, 2 mid/jun 3 high 4 college
	if(x['schoolprob'] > np.random.random()):
		for i in range(len(bracket)-1):
			if(x.age < bracket[i+1] and x.age >= bracket[i]):
				return i;
	return -1;


print("building outframe...")
out = pd.DataFrame();

out['id'] = indvs['id']
out['gender'] = indvs['gender'];
out['age'] = indvs['age']
out['schoolprob'] = out.apply(calcschoolprob, args=(agebracket, schoolprob),axis=1)
out['schoollevel'] = out.apply(assignschools,args=(gradebracket,),axis=1);

print("writing...");

out['id'] = out['id'].astype(int);
out = out.drop('age',axis=1);
out = out.drop('gender',axis=1);


#con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/employ-"+time.strftime("%Y-%m-%d_%H-%M-%S")+".sq3");
con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/school2.sq3");
out.to_sql('school',con);
con.close();