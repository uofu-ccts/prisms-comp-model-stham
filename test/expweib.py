from scipy.stats import exponweib;
import scipy.stats;
import scipy.optimize;
import matplotlib.pyplot as plt;
import matplotlib.cm as cm
import numpy as np;

#for a in np.arange(1.0,1, 1.0):
a = 1.0;
#cvals = np.linspace(1.0,3.14,10)
cvals = [1.0,1.5,1.8,2.1,3.34]
scvals = np.linspace(50.0,50.0,1)
cmapp = np.linspace(0.0,1.0,len(scvals));
colors = [ cm.jet(x) for x in cmapp ]


for i in range(len(scvals)): #np.arange(1.0,5.1, 0.5):
	sc = scvals[i]
	color = colors[i]
	for j in range(len(cvals)):
		c = cvals[j]
		x = np.linspace(0, exponweib.ppf(0.9999,a,c,scale=sc),1000)
		plt.plot(x,exponweib.pdf(x,a,c,scale=sc),color=color,alpha=0.5,label=str(c))

plt.ylim((0.0,0.028))
plt.xlim((0.0,200.0))
plt.legend();
plt.show()

exit()

def fexpweib(x,a,c,s,l):
	return exponweib.pdf(x,a,c,loc=l,scale=s);


bins = 1000;



a = np.linspace(1.0,1.0,bins);
c = np.linspace(1.0,1.0,bins);
sc = np.linspace(1.0,1.0,bins);
lc = np.linspace(0.0,0.0,bins);
lower = np.zeros(bins);
upper = np.zeros(bins);
median = np.zeros(bins);

for i in range(bins):
	result = exponweib.interval(0.99,a[i],c[i],loc=lc[i],scale=sc[i])
	upper[i] = result[1];
	lower[i] = result[0];
	median[i] = exponweib.median(a[i],c[i],loc=lc[i],scale=sc[i]);
	
plt.plot(lower,label='lower')
plt.plot(upper,label='upper')
plt.plot(median,label='median')
#plt.plot(a/c,label='ac')

plt.legend();
plt.show();



# abins = 75;
# cbins = 100;
# a = np.linspace(1.0,10.1,abins);
# c = np.linspace(1.0,10.1,cbins);
# upper = np.zeros((abins,cbins));
# lower = np.zeros((abins,cbins));
# median = np.zeros((abins,cbins));
# 
# for i in range(abins):
# 	for j in range(cbins):
# 		result = exponweib.interval(0.99,a[i],c[j],loc=0.0,scale=1.0)
# 		median[i][j] = exponweib.median(a[i],c[j],loc=0.0,scale=1.0);
# 		upper[i][j] = result[1]
# 		lower[i][j] = result[0];
# 
# 
# 
# 
# plt.matshow(upper);
# plt.colorbar();
# plt.show()
# plt.matshow(lower);
# plt.colorbar();
# plt.show();
# plt.matshow(median);
# plt.colorbar();
# plt.show();

exit();

r = exponweib.rvs(1.0,2.0,loc=0.0,scale=1.0,size=10);
interv = exponweib.interval(0.99,1.0,2.0,loc=0.0,scale=1.0)
print(interv);

binned,edges = np.histogram(r,bins, normed=True);

fit = exponweib.fit(r);

DF = bins - len(fit);
print(DF)
x = np.linspace(np.min(r),np.max(r),bins);
y = exponweib.pdf(x,*fit);

kern = scipy.stats.gaussian_kde(r)


res = binned - y;
s_err = np.sum(res**2)/DF
#err = exponweib.std(*fit);
#print(scipy.stats.pearsonr(binned,y), scipy.stats.spearmanr(binned,y));

ykern = kern(x)


#popt,pcov = scipy.optimize.curve_fit(fexpweib,x,ykern);

#print(popt)
#print(pcov)

fit2 = exponweib.fit(kern.resample(10000));
print(fit)
#fit2 = tuple(popt)


print(fit2)

plt.ylim((0.0,1.0))
xplot = np.linspace(interv[0],interv[1],100);
yactual = exponweib.pdf(xplot,1.0,2.0,loc=0.0,scale=1.0)
yplot = exponweib.pdf(xplot,*fit);
yplot2 = exponweib.pdf(xplot,*fit2);
#print(yplot2);

print(scipy.stats.spearmanr(yactual,yplot).correlation,scipy.stats.spearmanr(yactual,yplot2).correlation)

plt.hist(r,bins,normed=True);
plt.plot(xplot,yplot,label='fit');
plt.plot(xplot,yactual,label='actual');
plt.plot(xplot,kern(xplot),label='kernel');
plt.plot(xplot,yplot2,label='kernelfit');
#plt.plot(x,res,label='residues');
plt.legend()
plt.show();

