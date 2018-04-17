import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import time;
import datetime;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/bobwong/"
width = 900
# a = np.exp(np.linspace(10.0,1,width))[::-1]; a = a/np.max(a)

# a = np.linspace(1.0,0.0,width)
a = np.ones(width)
adjfact = 0.95
timeshift = 0.0

def getdatplot(name, deploy):

	df = pd.read_csv(datapath + name);

	df['utime'] = df['time'].apply(lambda x: np.datetime64(x))

	g1 = df.groupby('home_id');

	dep = g1.get_group(deploy);

	gdep = dep.groupby('entity_id');

	# monitorb003_humidity       408464
	# monitorb003_temperature    407169
	# monitorb003_large          407019
	# monitorb003_small          407019
	# Name: entity_id, dtype: int64
	


	c = 1
	#ax = plt.subplot(4,1,1);
	axprime = 0
	for ind,df in gdep:
		if(c == 1):	
			axprime = plt.subplot(4,1,c);
			axprime.set(ylabel=ind)
		else: 
			ax = plt.subplot(4,1,c,sharex=axprime);
			ax.set(ylabel=ind);
		# plt.plot(df['utime'], df['value'],label=name,linewidth=0.5,alpha=0.7)
		conv = np.convolve(df['value'].values,a,mode='same')
		cmax = np.max(conv)
		rmax = np.max(df['value'].values)
		# out = df['value'].values - (conv/cmax)*rmax
		# pos =  df['value'].diff().apply(lambda x: x if x > 0.0 else 0.0).cumsum().values
		# neg = df['value'].diff().apply(lambda x: np.abs(x) if x < 0.0 else 0.0).cumsum().values
		# pn = np.abs(pos-neg).cumsum()

		plt.plot(df['utime'], df['value'].values,label=name,linewidth=0.5,alpha=0.7)
		plt.plot(df['utime'].iloc[:len(conv)],conv/width,label=name,linewidth=0.5,alpha=0.7)
		plt.plot((df['utime'].iloc[0], df['utime'].iloc[-1]),(0.0,0.0),color='k',linewidth=0.5)
		# plt.plot(df['utime'], conv/cmax,label=name,linewidth=0.5,alpha=0.7)
		c+=1;



names = ['monitorb001.csv','monitorb002.csv','monitorb003.csv']
deploy = 'deployment_001'

# names = ['monitorb001.csv',]
# deploy = 'deployment_001'




# names = ['monitorb010.csv','monitorb011.csv','monitor110.csv']
# deploy = 'deployment_012'

# names = ['monitorb007.csv','monitorb009.csv','monitorb015.csv','monitorb016.csv','monitorb017.csv','monitorb019.csv','monitor110.csv']
# deploy = 'deployment_008'


for i in names:
	getdatplot(i, deploy);
	# break;

plt.show()
plt.legend()