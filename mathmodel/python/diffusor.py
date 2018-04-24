import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import datetime;
import h5py;

steplim = 96;

gridres = 100 #meters
kval = 0.5#fractional transport amount in terms of m/s (or m2/s or 1/s, depending on Ndims)
#can be thought of like a velocity (but it's not, it's really the
#fraction that passes through a hypothetical boundary plane)
dt = 15 # seconds
time = 15 #minute intervals for each step
halflife = 3600# halflife in seconds 

lamby = np.log(2)/3600
decay = np.exp(-lamby * dt)

def metaconvolve(a,sten):
	return np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=0,arr=a) + \
	np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=1,arr=a)

def iterate(arr, dep, steps, stencil):

	dep /= steps;
	
	for i in range(steps):
		arr += dep;
		newarr = metaconvolve(arr,stencil) + arr * decay ;
		arr = newarr

	return arr;

def runit():
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

	steps = (time * 60) // dt
	stencil = np.array([1,-2,1]) / 4.0
	stencil = stencil * kval / gridres * dt

	print("loading: ", datetime.datetime.now().time().isoformat())
	matfile = h5py.File(datapath + "Ftraj4-2018-03-19_13-53-28-24116.merge.sqlite3.h5")
	mats = []
	for i in range(0,steplim):
		mats += [matfile["/traj-slot-" + str(i).zfill(3) + "-set-002"][:]]
	matfile.close()

	#construct an initialization matrix by averaging all frames for the day
	#then iterating for 120 minutes
	print("initarr: ", datetime.datetime.now().time().isoformat())
	initmat = np.zeros_like(mats[0])
	for i in range(0,steplim):
		initmat += mats[i]
	initmat /= 96.0;
	initmat = iterate(initmat, np.zeros_like(initmat), steps * 8, stencil)

	#iterate using the proper depositions
	print("running: ", datetime.datetime.now().time().isoformat())
	outfile = h5py.File(datapath + "diffusedvals.h5")
	arr = initmat;
	for i in range(0,steplim):
		print(i, " ", datetime.datetime.now().time().isoformat(), flush=True)
		arr = iterate(arr,mats[i],steps, stencil);
		#it's probably wrong to use the final array as opposed to the intermediate arrays
		#but we can't use intermediate step arrays without really cranking up comp time
		ds = outfile.create_dataset("/traj-slot-"+str(i).zfill(3)+"-set-002",data=arr,fillvalue=0.,compression='gzip',compression_opts=9)
	outfile.close();
	print("finished: ", datetime.datetime.now().time().isoformat())

if __name__ == "__main__":
	runit();




