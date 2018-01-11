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
# print(np.max(rawmat))
#exit()

# hist,edge = np.histogram(rawmat,bins=30)
# print(edge,hist)
# plt.plot(edge[:-1],hist)
# plt.show()
# maxval = np.max(rawmat)*0.5
# maxval = 1500.0



#print(rawmat.shape);
rawmat = np.swapaxes(rawmat,2,3)
#print(rawmat.shape);
# rawmat = rawmat / np.max(rawmat)


# rawmat = np.log10(rawmat + 1);
maxval = np.max(rawmat)
rawmat = np.clip(rawmat, 0.0,maxval);
print(maxval)

rawmat[:,2,:,:] *= 10;

#wasatch front
# xlim = (100,600)
# ylim = (600,1100)

#slc county
xlim = (315,415)
ylim = (750,850)

c = 1;
s = 0   # starting hour
e = 23  # ending hour
st = 2; # stepsize

plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

for j in range(0,3):
	for i in range(s,e + 1,st):
		ax = plt.subplot(3,(e - s + 1)//st,c)
		ax.set_aspect(1.0)
		ax.axis('off')
#		ax.invert_xaxis()
		ax.set_xlim(xlim);
		ax.set_ylim(ylim);
		im = ax.pcolormesh(rawmat[i,j],vmin=0,vmax=maxval,cmap=plt.get_cmap('viridis'))
		#plt.colorbar(im,ax=ax)
		c+= 1


F = plt.gcf();
F.set_size_inches(10,5)
F.set_dpi(300.0);
F.savefig(datapath +str(sys.argv[1])+"-plot.png",dpi=300);

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

#plt.show();


infile.close();