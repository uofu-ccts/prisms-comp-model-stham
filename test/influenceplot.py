import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime
import sys;
import h5py;
import mkl;
import multiprocessing as mp;
from itertools import repeat;



def plot(i,path,vmax):
# for i in range(0,96):
	diffile = h5py.File(path,'r')
	for j in range(3):
		submat = diffile["/traj-slot-"+str(i).zfill(3)+"-set-"+str(j).zfill(3)][:][:].T
		submat[submat < 0.0] = 0.0;
		submat[submat > 0.0] += (vmax - submat[submat > 0.0]) * 0.05
		ax = plt.axes([0.,0.,1.,1.])
		ax.set_axis_off()
		ax.set_aspect(1.0)
		ax.set_ylim(1400,3000)
		ax.set_xlim(1000,2600)
		ax.pcolormesh(submat,vmin=0.0,vmax=vmax);
		# ax.pcolormesh(submat,vmin=0.0);
		ax.text(0.05,0.05,str(((i//4)+4)%24).zfill(2)+":"+str(15*(i%4)).zfill(2),color='w',fontsize=18,transform=ax.transAxes)
		F=plt.gcf()
		F.set_size_inches(8,8)
		# F.set_dpi(200.0)
		F.savefig("outfigraw-slot-"+str(i).zfill(2)+"-set-"+str(j).zfill(3)+".png",dpi=200)
		plt.clf()

	diffile.close();


def runit(threads):
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	# fname = "newdiffusedvals.h5"
	fname = "Ftraj4-2018-04-25_17-20-10-ForkPoolWorker-10.merge.sqlite3.h5"

	# diffile = h5py.File(datapath + fname,'r')
	
	# for i in range(96):
	# 	for j in range(3):
	# 		vmax = np.maximum(np.max(diffile["/traj-slot-"+str(i).zfill(3)+"-set-"+str(j).zfill(3)][:][:]),vmax)
	# print(vmax)
	vmax = 50.00
	# diffile.close()

	p = mp.Pool(threads);
	p.starmap(plot,zip(np.arange(36,37),repeat(datapath + fname),repeat(vmax)),chunksize=12);
	p.close();



if __name__ == "__main__":
	threads = mkl.get_max_threads();
	threads = 8;
	runit(threads);

# # outfile = h5py.File(datapath + "influence-comp.h5")
# # ds = outfile.create_dataset('populations',data=rawmat,fillvalue=0.,compression='gzip',compression_opts=9)
# # ds.attrs['xorigin'] = xorigin ;
# # ds.attrs['yorigin'] = yorigin ;
# # ds.attrs['day'] = 3;
# # #ds.attrs['date']=str(
# # ds.attrs['grid']=500.0
# # outfile.close();
# # print(np.max(rawmat))
# #exit()

# # hist,edge = np.histogram(rawmat,bins=30)
# # print(edge,hist)
# # plt.plot(edge[:-1],hist)
# # plt.show()
# # maxval = np.max(rawmat)*0.5
# # maxval = 1500.0



# #print(rawmat.shape);
# rawmat = np.swapaxes(rawmat,2,3)
# #print(rawmat.shape);
# # rawmat = rawmat / np.max(rawmat)


# # rawmat = np.log10(rawmat + 1);
# maxval = np.max(rawmat)
# rawmat = np.clip(rawmat, 0.0,maxval);
# print(maxval)

# rawmat[:,2,:,:] *= 10;

# #wasatch front
# # xlim = (100,600)
# # ylim = (600,1100)

# #slc county
# xlim = (315,415)
# ylim = (750,850)

# c = 1;
# s = 0   # starting hour
# e = 23  # ending hour
# st = 2; # stepsize

# plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)

# for j in range(0,3):
# 	for i in range(s,e + 1,st):
# 		ax = plt.subplot(3,(e - s + 1)//st,c)
# 		ax.set_aspect(1.0)
# 		ax.axis('off')
# #		ax.invert_xaxis()
# 		ax.set_xlim(xlim);
# 		ax.set_ylim(ylim);
# 		im = ax.pcolormesh(rawmat[i,j],vmin=0,vmax=maxval,cmap=plt.get_cmap('viridis'))
# 		#plt.colorbar(im,ax=ax)
# 		c+= 1


# F = plt.gcf();
# F.set_size_inches(10,5)
# F.set_dpi(300.0);
# F.savefig(datapath +str(sys.argv[1])+"-plot.png",dpi=300);

# # ax = plt.subplot(1,3,2)
# # ax.invert_xaxis()
# # ax.set_xlim(xlim);
# # ax.set_ylim(ylim);
# # ax.pcolormesh(rawmat[10,1],cmap=plt.get_cmap('viridis'))
# # 
# # ax = plt.subplot(1,3,3)
# # ax.invert_xaxis()
# # ax.set_xlim(xlim);
# # ax.set_ylim(ylim);
# # ax.pcolormesh(rawmat[10,2],cmap=plt.get_cmap('viridis'))

# #plt.show();


# infile.close();