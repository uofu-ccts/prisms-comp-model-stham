import pandas as pd;
import numpy as np;
import pyproj;
import sqlite3;

def traj(n,agentnum,day,t):
	df = pd.DataFrame({"agentnum":np.zeros((n))});
	df["agentnum"] = agentnum;
	df["day"] = day;
	df["day365"] = np.random.randint(0,365);
	df["start"] = np.sort(np.random.randint(0,1440,size=n));
	df["start"].iloc[0] = 0;
	df["length"] = df["start"].shift(-1) - df["start"];
	df["length"].iloc[-1] = 1440 - df["start"].iloc[-1];
	df["length"] = df["length"].astype(np.int32);

	df["actcode"] = np.random.randint(0,1000000,size=n);

	df["lat"] = np.random.randn(n)*100000.0+np.random.randint(4500000,4600000);
	df["long"] = np.random.randn(n)*10000.0+np.random.randint(400000,410000);
	
	df[["lat","long"]] = df[["lat","long"]].apply(t,axis=1);

	return df;


outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');
def t(x):
	x.long,x.lat=pyproj.transform(inproj,outproj,x.long,x.lat);
	return x
	# return (x.locx+1,x.locy+1);


out = pd.DataFrame();

for i in range(10):
	for j in range(1,8):
		out = pd.concat([out,traj(np.random.randint(10,40),i,j,t)])

con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/acttraj.sq3")
out.to_sql("acttraj",con, index=False);
con.close();