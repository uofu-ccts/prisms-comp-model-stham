import pandas as pd;
import time;
import scipy;
import scipy.stats;
import numpy as np;
import sqlite3;
# import matplotlib.pyplot as plt;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

print("reading...")
acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
infotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
print("joining...")
jointable = pd.merge(acttable,infotable,on='TUCASEID')


mapping = np.sort(list(set(jointable['TRCODE'])))
count = len(mapping)

g = jointable.groupby(['TRCODE'])

#tri = { tr:i for i,tr in enumerate(mapping) }
#itr = { i:tr for i,tr in enumerate(mapping) }

print('processing...')
#parse the table for activity lengths and calc wiebull params
data = np.zeros((count,6));

for i in range(count):
	a = g.get_group(mapping[i])['TUACTDUR']
	w = scipy.stats.exponweib.fit(a, fa=1.0,floc=0.0);
	
	#D,p = scipy.stats.kstest(a,'exponweib',w,alternative='greater')
	
	#print(mapping[i], D, p, len(a));
	#print(w);
	#data[i] = (mapping[i],w[1],w[3], len(a), D, p)
	data[i] = (mapping[i],w[1],w[3], len(a))


# plt.plot(data[:,3],data[:,4], '.')
# plt.show();

#exit()

#write to table
print('writing...')


frame = pd.DataFrame(data, columns=['ID','c','scale','n','ksD','ksp']);

con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/weibull.sq3");
frame.to_sql('weibull',con);
con.close();