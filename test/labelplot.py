import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

pd.set_option('display.max_rows', 1000)

def labelReduce(labels):
	labelset = set(labels);
	labelmap = { item:index for index,item in enumerate(labelset) }
	print(labelmap)
	newlabels = np.array([labelmap[b] for b in labels]);
	return newlabels;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/test/"

labels = pd.read_csv(datapath+"2017-03-02_17-41-57/labels.csv")

#labels['finallabels']=labelReduce(labels['finallabels'].values);
#labels['secondlabelreduce']=labelReduce(labels['secondlabelreduce'].values);

g = labels.groupby(["finallabels","secondlabelreduce"])
c = g.finallabels.count()
print(c)

xy,z = zip(*c.iteritems())

x,y = zip(*xy)



z_arr = np.zeros((max(x)+1,max(y)+1))
x = np.array(x)
y = np.array(y)
z = np.array(z)
z_arr[x,y]=z

z_arr = z_arr.T

z_arr[z_arr == 0] = np.NaN;

ax1 = plt.subplot(1,2,1)
im1 = ax1.matshow(z_arr,cmap='viridis_r')
plt.colorbar(im1);
ax2 = plt.subplot(1,2,2)
im2 = ax2.matshow(np.log10(z_arr),cmap='viridis_r')
plt.colorbar(im2)
print("Percent same labels ", np.sum(labels['finallabels']==labels['secondlabelreduce'])/len(labels)*100)
plt.show()

