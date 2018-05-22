import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import datetime;
import h5py;
import multiprocessing as mp;
import mkl;
from itertools import repeat;

steplim = 96;

gridres = 100 #meters
kval = 0.5#fractional transport amount in terms of m/s (or m2/s or 1/s, depending on Ndims)
#can be thought of like a velocity (but it's not, it's really the
#fraction that passes through a hypothetical boundary plane)
dt = 15 # seconds
time = 15 #minute intervals for each step
halflife = 3600# halflife in seconds 

lamby = np.log(2)/halflife
decay = np.exp(-lamby * dt)


def convwrap(x,sten):
	return np.convolve(x,sten,mode='same');

def altwrap(a,sten):
	return np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=0,arr=a)



def parallelmetaconv(a,sten,pool):

	s = pool._processes
	out = pool.starmap(altwrap, zip(np.array_split(a,s),repeat(sten)))

	ind = 0;
	for i in range(len(out)):
		a[ind:ind+len(out[i])] = out[i];
		ind += len(out[i])
	a = a.T

	out = pool.starmap(altwrap, zip(np.array_split(a,s),repeat(sten)))

	ind = 0;
	for i in range(len(out)):
		a[ind:ind+len(out[i])] = out[i];
		ind += len(out[i])

	return a.T


def metaconvolve(a,sten):
	return np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=0,arr=a) + \
	np.apply_along_axis(lambda x: np.convolve(x,sten,mode='same'),axis=1,arr=a)

def iterate(arr, dep, steps, stencil, pool=None):

	dep /= steps;
	
	for i in range(steps):
		arr += dep;
		if(pool is None):
			newarr = metaconvolve(arr,stencil) + arr * decay ;
		else: 
			newarr = parallelmetaconv(arr,stencil,pool) + arr * decay ;
		arr = newarr

	return arr;

def runit(threads):
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

	steps = (time * 60) // dt
	stencil = np.array([1,-2,1]) / 4.0
	stencil = stencil * kval / gridres * dt


	# pool = mp.Pool(threads);
	pool = None

	outfile = h5py.File(datapath + "newdiffusedvals.h5")
	for n in range(3):
		print("loading: ", datetime.datetime.now().time().isoformat())
		matfile = h5py.File(datapath + "Ftraj4-2018-04-25_17-20-10-ForkPoolWorker-10.merge.sqlite3.h5")
		mats = []
		for i in range(0,steplim):
			mats += [matfile["/traj-slot-" + str(i).zfill(3) + "-set-"+str(n).zfill(3)][:]]
		matfile.close()

		#construct an initialization matrix by averaging all frames for the day
		#then iterating for 120 minutes
		print("initarr: ", datetime.datetime.now().time().isoformat())
		initmat = np.zeros_like(mats[0])
		for i in range(0,steplim):
			initmat += mats[i]
		initmat /= 96.0;
		initmat = iterate(initmat, np.zeros_like(initmat), steps * 8, stencil, pool)

		#iterate using the proper depositions
		print("running: ", datetime.datetime.now().time().isoformat())
		
		arr = initmat;
		for i in range(0,steplim):
			print(i, " ", datetime.datetime.now().time().isoformat(), flush=True)
			arr = iterate(arr,mats[i],steps, stencil, pool);
			#it's probably wrong to use the final array as opposed to the intermediate arrays
			#but we can't use intermediate step arrays without really cranking up comp time
			ds = outfile.create_dataset("/traj-slot-"+str(i).zfill(3)+"-set-"+str(n).zfill(3),data=arr,fillvalue=0.,compression='gzip',compression_opts=9)
			ds.flush()

	outfile.close();
	print("finished: ", datetime.datetime.now().time().isoformat())

if __name__ == "__main__":
	threads = mkl.get_max_threads();
	threads = 8;
	runit(threads);




