import matplotlib.pyplot as plt;
import numpy as np;
import h5py;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run03/"
path = path = datapath + "newdiffusedvals.h5"

infile = h5py.File(path,'r')

def getstick(x,y,file):                         
	vals = np.zeros((3,96))
	for i in range(0,3):
		for j in range(0,96):
			vals[i][j] = file["/traj-slot-"+str(j).zfill(3)+"-set-"+str(i).zfill(3)][x,y]
	return vals;


samples = 10

x = np.random.randint(1970,2240,size=samples)
y = np.random.randint(1800,1945,size=samples)


styles = ['-','--',':']
cols = ['k','b','r']

for i in range(samples):
	v = getstick(x[i],y[i],infile)


	for j in range(0,3):
		plt.plot(v[j],linestyle=styles[j],color=cols[j],linewidth=0.75);

plt.show()

infile.close();