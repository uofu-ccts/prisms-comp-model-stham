import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime
import sys;
import h5py;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

#infile = h5py.File(datapath + "influence-comp.h5", 'r')
infile = h5py.File(sys.argv[1],'r')
xorigin = infile['populations'].attrs['xorigin']
yorigin = infile['populations'].attrs['yorigin']
rawmat = infile['populations'][:24,:3,:,:];


# outfile = h5py.File(datapath + "influence-comp.h5")
# ds = outfile.create_dataset('populations',data=rawmat,fillvalue=0.,compression='gzip',compression_opts=9)
# ds.attrs['xorigin'] = xorigin ;
# ds.attrs['yorigin'] = yorigin ;
# ds.attrs['day'] = 3;
# #ds.attrs['date']=str(
# ds.attrs['grid']=500.0
# outfile.close();
print(np.max(rawmat))
#exit()

#print(rawmat.shape);
rawmat = np.swapaxes(rawmat,2,3)
#print(rawmat.shape);
#rawmat = np.clip(rawmat, 0.0,1000.0);
rawmat = np.log10(rawmat + 1);

#xlim = (750,950)
#ylim = (600,1100)

c = 1;
s = 12
e = 23

for j in range(0,3):
	for i in range(s,e + 1):
		ax = plt.subplot(3,e - s + 1,c)
		ax.set_aspect(1.0)
		ax.axis('off')
#		ax.invert_xaxis()
		#ax.set_xlim(xlim);
		#ax.set_ylim(ylim);
		ax.pcolormesh(rawmat[i,j],cmap=plt.get_cmap('viridis'))
		c+= 1

# ax = plt.subplot(1,3,2)
# ax.invert_xaxis()
# ax.set_xlim(xlim);
# ax.set_ylim(ylim);
# ax.pcolormesh(rawmat[10,1],cmap=plt.get_cmap('viridis'))
# 
# ax = plt.subplot(1,3,3)
# ax.invert_xaxis()
# ax.set_xlim(xlim);
# ax.set_ylim(ylim);
# ax.pcolormesh(rawmat[10,2],cmap=plt.get_cmap('viridis'))

plt.show();


infile.close();