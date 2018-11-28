import matplotlib.pyplot as plt;
import numpy as np;
import h5py;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run05/"
path = path = datapath + "diffusedvals.h5"

infile = h5py.File(path,'r')

def getstick(x,y,file):                         
	vals = np.zeros((3,96))
	for i in range(0,3):
		for j in range(0,96):
			print(i,",",j,flush=True,end=' ')
			vals[i][j] = np.sum(file["/traj-slot-"+str(j).zfill(3)+"-set-"+str(i).zfill(3)])
	return vals;


samples = 10

# x = np.random.randint(1970,2240,size=samples)
# y = np.random.randint(1800,1945,size=samples)


styles = ['-','--',':']
cols = ['k','b','r']

# for i in range(samples):
v = getstick(0,0,infile)

allsum = np.sum(v,axis=0)
v = v/allsum


for j in range(0,3):
	plt.plot(v[j],linestyle=styles[j],color=cols[j],linewidth=0.75);

plt.show()

infile.close();