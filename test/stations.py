import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import datetime;


stationspath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/stations/"

statlist = pd.read_csv(stationspath + "statlocs",skiprows=[3,])

df = {}
print("load: ", datetime.datetime.now().time().isoformat())
c = 0;
for i in statlist["station"].values :
	df[i] = pd.read_csv(stationspath + i +".csv",skiprows=6).dropna()
	df[i]["PM_25_concentration_set_1"] = pd.to_numeric(df[i]["PM_25_concentration_set_1"])
	df[i]["Date_Time"] = pd.to_datetime(df[i]["Date_Time"])
	# plt.plot(df[i]["Date_Time"],df[i]["PM_25_concentration_set_1"]+c,linewidth=0.5,label=str(i))
	# c+=50


# plt.legend()
# plt.show()

# sample = df['QHW']["Date_Time"].sample(n=10);

print("hist: ",datetime.datetime.now().time().isoformat())

keys = list(df.keys());

hist = []
deltahist = []
acchist = []

for i in range(len(keys)):
	for j in range(i+1,len(keys)):
		# if(i == j): continue;
		print(keys[i],keys[j])
		m = pd.merge(df[keys[i]],df[keys[j]],how='inner',on=['Date_Time','Date_Time']);
		# print(m.columns)
		if 'PM_25_concentration_set_1_x' in m.columns:
			arr = (m['PM_25_concentration_set_1_x']-m['PM_25_concentration_set_1_y'])
			if(keys[i] == "QHW" or keys[j] == "QHW"):
				plt.plot(m["Date_Time"], arr.diff(),linewidth=0.75,label=keys[i]+','+keys[j]);
			print(np.mean(arr),np.std(arr))
			# plt.scatter(np.max(np.array([m['PM_25_concentration_set_1_x'],m['PM_25_concentration_set_1_y']]).T,axis=1),np.abs(arr),s=3)
			# if(np.sum(arr) < 0.0): arr = np.log10(m['PM_25_concentration_set_1_y']/m['PM_25_concentration_set_1_x'])
			# h,b = np.histogram(arr, bins=101,range=(-3,3),normed=True)
			h,b = np.histogram(arr, bins=201,range=(-100,100),normed=True)
			hist += [[keys[i],keys[j],h]]
			
			darr = np.log10(m['PM_25_concentration_set_1_x'].diff()/m['PM_25_concentration_set_1_y'].diff())
			# if(np.sum(arr) < 0.0): arr = np.log10(m['PM_25_concentration_set_1_y']/m['PM_25_concentration_set_1_x'])
			dh,b = np.histogram(darr, bins=101,range=(-3,3),normed=True)
			deltahist += [[keys[i],keys[j],dh]]

			d2arr = np.log10(m['PM_25_concentration_set_1_x'].diff().diff()/m['PM_25_concentration_set_1_y'].diff().diff())
			# if(np.sum(arr) < 0.0): arr = np.log10(m['PM_25_concentration_set_1_y']/m['PM_25_concentration_set_1_x'])
			d2h,b = np.histogram(d2arr, bins=101,range=(-3,3),normed=True)
			acchist += [[keys[i],keys[j],d2h]]

# 	print(df[i][df[i]["Date_Time"].isin(sample)])

print("plot: ", datetime.datetime.now().time().isoformat())

# print(hist[0])

for i in hist:
	# plt.plot(np.power(10,np.linspace(-3,3,101)),i[2],label=i[0]+","+i[1],linewidth=0.75)
	plt.plot(np.linspace(-100,100,201),i[2]/np.sum(i[2]),label=i[0]+","+i[1],linewidth=0.75)

plt.legend();
plt.show();

# for i in deltahist[0:5]:
# 	# plt.plot(np.power(10,np.linspace(-3,3,101)),i[2],label=i[0]+","+i[1],linewidth=0.75)
# 	plt.plot(np.linspace(-3,3,101),i[2],label=i[0]+","+i[1],linewidth=0.75)

# plt.legend();
# plt.show();

# for i in acchist[0:5]:
# 	# plt.plot(np.power(10,np.linspace(-3,3,101)),i[2],label=i[0]+","+i[1],linewidth=0.75)
# 	plt.plot(np.linspace(-3,3,101),i[2],label=i[0]+","+i[1],linewidth=0.75)

# plt.legend();
# plt.show();