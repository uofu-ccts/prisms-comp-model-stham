import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rollplot(path, color):
	ad = pd.read_csv(path)
	ad['utime'] = pd.to_datetime(ad['Date'])
	g = ad[['utime','Daily Mean PM2.5 Concentration']].groupby('utime')
	# for i in [(0.,'_'),(.25,'2'),(.50,'.'),(.75,'1'),(1.00,'_')]:
	for i in [(0.,'_'),(.50,'.'),(1.00,'_')]:
		plt.plot(g.quantile(i[0]),linewidth=0.0,marker=i[1],color=color,alpha=0.75)
	for ind,df in g:
		plt.plot((ind,ind),(df['Daily Mean PM2.5 Concentration'].min(),df['Daily Mean PM2.5 Concentration'].max()),linewidth=0.75,color=color,alpha=0.7)

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
files = [ "ad_viz_plotval_data-chicagometro.csv", "ad_viz_plotval_data-wdcmetro.csv","ad_viz_plotval_datadallasmetro.csv"]
colors = [ 'r','b','g']
z = zip(files,colors)
for i in z:
	rollplot(i[0],i[1])


plt.show()

