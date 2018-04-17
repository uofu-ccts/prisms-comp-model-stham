import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from scipy.signal import convolve;
import datetime;


def metaconvolve(a,sten):
	return np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=0,arr=a) + \
	np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=1,arr=a)


# from scipy.stats import beta

gridres = 100 #meters
area = 400000 #meters
# pollpos = [1000,] #meters from center of zero
pollpos = [[10000//gridres,50000//gridres],[10000//gridres,50000//gridres]]

kvals = [0.5,] #fractional transport amount in terms of m/s (or m2/s or 1/s, depending on Ndims)
#can be thought of like a velocity (but it's not, it's really the
#fraction that passes through a hypothetical boundary plane)
pollution = 10.0 #kg/minute
pollstdev = 0.0 #kg/minute
dt = 60 # seconds
minutes = 120
time = 60*minutes #seconds
record = 60*15; #seconds

steps = time // dt
recst = record // dt
print("Steps:",steps)


# x = np.linspace(0,area,area/gridres)
# pos = np.array([np.int(np.floor(npl.array(pollpos) / gridres))])
pos = pollpos
pdelta = pollution / 60.0
pstd = pollstdev / 60.0

for k in kvals:

	stencil = np.array([1,-2,1]) / 4.0
	stencil = stencil * k / gridres * dt
	print(stencil)

	arr = np.zeros((area//gridres,area//gridres))
	

	arr = np.abs(np.random.randn(area//gridres,area//gridres))* pstd + pdelta
	vmax = np.max(arr)
	# arr[pos] = 1.0
	print(np.sum(arr))
	total = 0.0
	print(datetime.datetime.now().time().isoformat());
	for i in range(steps):
		# arr[pos] += np.abs(np.random.randn(len(pos[0]))) * pstd + pdelta * dt
		# arr[pos] += pdelta
		if(i % recst == 0):
			print(i,end=" ",flush=True);
			# plt.matshow(arr,vmax=vmax);plt.show();
			# plt.plot(x,arr, linestyle='-',linewidth=0.5)

		newarr = metaconvolve(arr,stencil) + arr

		arr = newarr
	print(datetime.datetime.now().time().isoformat());
	print(np.sum(arr))
	# plt.plot(x,arr, linestyle='-',linewidth=0.5)
	plt.matshow(arr,vmax=vmax);
plt.show();




# x = np.linspace(0,area,area/gridres)
# pos = np.array([np.int(np.floor(np.array(pollpos) / gridres))])
# pdelta = pollution / 60.0
# pstd = pollstdev / 60.0

# for k in kvals:

# 	stencil = np.array([1,-2,1]) / 2.0
# 	stencil = stencil * k / dt

# 	arr = np.zeros_like(x)

# 	arr[pos] = 1.0

# 	total = 0.0
# 	for i in range(steps):
# 		# arr[pos] += np.abs(np.random.randn(len(pos))) * pstd + pdelta
# 		# arr[pos] += pdelta
# 		if(i % recst == 0):
# 			plt.plot(x,arr, linestyle='-',linewidth=0.5)

# 		newarr = np.convolve(arr,stencil,mode='same') + arr

# 		arr = newarr
# 	print(np.sum(arr))
# 	plt.plot(x,arr, linestyle='-',linewidth=0.5)
# plt.show();

