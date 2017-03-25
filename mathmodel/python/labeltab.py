import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

def arrayify(table):
	g = labels.groupby(["casetypelabelreduce","daytypelabelreduce"])
	c = g.casetypelabelreduce.count()
	print(c)
	xy,z = zip(*c.iteritems())
	x,y = zip(*xy)
	z_arr = np.zeros((max(x)+1,max(y)+1))
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	z_arr[x,y]=z
	
	return z_arr;


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/final-label-classifier"

labels = pd.read_csv(datapath+"labels.csv")

weekend = labels[labels['TUDIARYDAY'].apply(lambda x: x in (1,7))]
weekday = labels[labels['TUDIARYDAY'].apply(lambda x: x in (2,3,4,5,6))]

weekendarr = arrayify(weekend);
weekdayarr = arrayify(weekday);

ax1 = plt.subplot(1,2,1)
im1 = ax1.matshow(weekdayarr,cmap='viridis_r')
plt.colorbar(im1);
ax2 = plt.subplot(1,2,2)
im2 = ax2.matshow(weekendarr,cmap='viridis_r')
plt.colorbar(im2)
plt.show()


