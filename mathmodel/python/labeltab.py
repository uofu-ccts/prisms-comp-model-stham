import pandas as pd;
import numpy as np;
import h5py;
# import matplotlib.pyplot as plt;

def arrayify(table):
	g = table.groupby(["casetypelabelreduce","daytypelabelreduce"])
	c = g.casetypelabelreduce.count()
	xy,z = zip(*c.iteritems())
	x,y = zip(*xy)
	z_arr = np.zeros((max(x)+1,max(y)+1))
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	z_arr[x,y]=z
	for i in range(len(z_arr)):
		z_arr[i] = z_arr[i] / np.sum(z_arr[i])
	return z_arr;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/final-label-classifier/"

labels = pd.read_csv(datapath+"labels.csv")

weekend = labels[labels['TUDIARYDAY'].apply(lambda x: x in (1,7))]
weekday = labels[labels['TUDIARYDAY'].apply(lambda x: x in (2,3,4,5,6))]

weekendarr = arrayify(weekend);
weekdayarr = arrayify(weekday);

out = h5py.File(datapath + "labeltab.h5");
ds = out.create_dataset("/weekend",data=weekendarr,fillvalue=0.,compression='gzip',compression_opts = 9);
ds = out.create_dataset("/weekday",data=weekdayarr,fillvalue=0.,compression='gzip',compression_opts = 9);
out.close();

# ax1 = plt.subplot(1,3,1)
# im1 = ax1.matshow(weekdayarr,cmap='Greys')
# plt.colorbar(im1);
# ax2 = plt.subplot(1,3,2)
# im2 = ax2.matshow(weekendarr,cmap='Greys')
# plt.colorbar(im2)
# ax3 = plt.subplot(1,3,3)
# im3 = ax3.matshow(weekdayarr-weekendarr,cmap='seismic')
# plt.colorbar(im3)
# plt.show()


