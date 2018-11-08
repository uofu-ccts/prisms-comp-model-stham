import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from collections import Counter;

#enum for ptable
ACTCOUNT = 0
LENCOUNT = 1
LENACTJOINTPROB = 2
ACTWINS = 3
LENWINS = 4
LHIST = 5
LBINS = 6
LHISTLEN = 7
ORDERPROB = 8

def genAgents(n):
	pass;

def genActivities(n_acts, n_types,n_loctypes,contextlam = 3, maxlen = 0.5, maxlenab = 0.7, minload = 2.0,n_samples = 100):


	contexts = np.random.poisson(contextlam,size=n_acts)+1
	m = np.sum(contexts);
	counts = Counter({ind:c for ind,c in enumerate(contexts)})
	maincode = np.array(list(counts.elements()))
	mainfreq = np.random.pareto(2, size=m)
	mainfreq = mainfreq/(np.max(mainfreq)*1.01)

	print(m)

	#length base distros
	alpha_len = np.power(10,(np.random.randn(m)/2.0))
	beta_len = np.power(10,(np.random.randn(m)/2.0))
	maxlens = np.random.beta(maxlenab,maxlenab,size=m)*maxlen;


	#start base distros
	alpha_st = np.power(10,(np.random.randn(m)/2.0))
	beta_st = np.power(10,(np.random.randn(m)/2.0))

	loadweights = maxlens * mainfreq

	ptable = []

	for i in range(n_types):
		picks = np.full(m,False);
		for j in range(5):
			picks = picks | (np.random.random(m) < mainfreq)
			load = np.sum(loadweights[picks])
			if (load > minload): break;
		picks = np.nonzero(picks)[0]
		print(load,len(picks))
		print(picks)
		samples = np.zeros((3,n_samples,len(picks))) #start, len, freq
		pickprob = np.random.pareto(1, size=len(picks))
		pickprob = pickprob/(np.max(pickprob)*1.01)
		for j in range(n_samples):
			samples[0][j][:] = np.random.beta(alpha_st[picks],beta_st[picks])
			samples[1][j][:] = np.random.beta(alpha_len[picks],beta_len[picks])
			samples[2][j][:] = (np.random.random(len(picks)) < pickprob ).astype(np.float64)
			
		

		

		if(i > 3) : break;


	#who - self,hh,affiliate, what fraction

	#when - typical activity length and window (fixed or flexible)

	#where - type for each context

	#what - essential, demographically essential or elective, frequency halflife

	#how - which context



	




def genDemographics(n_demos,n_types):

	pass;
	


def affexp(x,aff):
	k = np.power(2,aff)
	return np.power(2,x*k)/np.power(2,k)


def sqbox(t):
	cs = np.cos(t); sn = np.sin(t);
	div = np.maximum(np.abs(cs),np.abs(sn))
	return cs / div, sn / div


def rloc(x,y,r,setback=25):
	scale = (r-setback)/r
	s = np.random.random() * 2 * np.pi
	cs = np.cos(s); sn = np.sin(s);
	div = max(np.abs(cs),np.abs(sn))
	return (cs/div)*r*scale*0.5 + (x+0.5)*r, (sn/div)*r*scale*0.5 + (y+0.5)*r

def adjacent(blocks):
	# roadmask = blocks == np.max(blocks)
	mask = blocks >=0
	out = np.zeros_like(blocks)
	out[mask] = 1
	# out[roadmask] = 0
	stencil = np.array([1,1,1])
	out = np.apply_along_axis(lambda x: np.convolve(x,stencil,mode='same'),axis=0,arr=out).T
	out = np.apply_along_axis(lambda x: np.convolve(x,stencil,mode='same'),axis=0,arr=out).T
	out[mask] = 0
	# out[roadmask] = 0;
	return np.nonzero(out > 0)


def genlocs(n = 1000, blocksize = 10,blockwidth = 100, majorroad = 10, density = 100, verticalthresh = 0.95, mode='single', weights = None, affinity = 1.0, seed = None, types = None,emptyselect = 0.20,totalrand=0.01):
	""" 
	Creates an artificial list of locations
	n - total number of locations to populate
	blocksize - width of region in blocks
	blockwidth - size of a block, meters
	majorroad - number of blocks before a major roadway is built
	density - maximum number of locations per block before vertical scaling test occurs
	verticalprob - probability of additional vertical units being added at capacity
	mode - single or inclusive zoning method
	weights - relative weights of location types
	affinity - affinity for an already populated block. affinity is the log of k, and should be between (-4,4) for good results
	seed - RNG seed for the loc


	
	Locations are populated on a flat plane with meter lcoations.

	All locations are assigned to a block grid with minor roads between blocks.
	Block are allocated to construct major roads which are preferred for routing
	Locations can be assigned to a grid until they reach the maximum density
	When the maximum density is reached on a location a probability test is performed
	to see if additional vertical units will be added to the block.  
	
	Locations are assigned to zones in the same way zoning laws define landuse. That is, 
	blocks have designated compositions that are preferable. There are two views 
	into this, the US view, and the Japanese view, where the US is a single use zone
	and the Japanese view is inclusive zones (meaning zones have a maximum permissible
	nuisance level. 

	The zone types in this simple model are Residential, Business, Commercial, Industrial,
	Institutional, Open, and Roadway. Roadways are -always- roadways, no mixing. Industrial activities are always mututally exclusive with institutional and residential zones. Open zones are also exclusive to other types. 


	"""

	if(seed != None):
		np.random.seed(seed);
		print("RNG:", seed)

	if(types == None):
		types = ['Residential','Business','Commercial','Industrial','Institutional','Open']
		#major roadway is implicitly last number
	typecount = len(types)

	# radius = n / density
	# blocksize = int(np.floor(2*radius))+1
	# blocksize += 1

	blocks = np.zeros((3,blocksize,blocksize)) #type,count,max,p


	if(weights == None):
		weights = [0.4,0.15,0.15,0.15,0.05,0.10] #weights of types

	locs = np.zeros((3,n)) #type, x, y

	
	blocks[0] = -1
	blocks[2] = density				

	
	locs[0] = np.random.choice(typecount,p=weights,size=n)
	locs[0][np.arange(typecount)] = np.arange(typecount);

	for i in range(n):



		maxdense = np.max(blocks[2])
		s = 0;p = 0
		if mode == 'single':
			select = np.nonzero(blocks[0] == locs[0][i])
		elif mode == 'inclusive':
			select = np.nonzero(blocks[0] >= locs[0][i])

		empty = np.nonzero(blocks[0] == -1)
		emptyadj = adjacent(blocks[0])
		
		
		# totalrand
		if (len(empty[0]) > 0 and np.random.random() < totalrand) or i == 0:
			if(len(select[0]) > 0):
				s = np.append(select,empty,axis=1)
			else:
				s = empty;
			p = np.ones_like(s[0])
		# emptyselect
		elif(len(empty[0]) > 0 and np.random.random() < emptyselect) or i < typecount:
			s = emptyadj
			p = np.ones_like(s[0])
		# affinity
		else:
			s = select
			p = affexp( blocks[1][s[0],s[1]] / maxdense ,affinity)


		#select block
		pick = np.random.choice(len(p),p=p/np.sum(p))
		locs[1][i],locs[2][i] = rloc(s[0][pick],s[1][pick],blockwidth)
		blocks[1][s[0][pick],s[1][pick]] += 1
		if(blocks[0][s[0][pick],s[1][pick]] == -1):
			blocks[0][s[0][pick],s[1][pick]] = locs[0][i]

		#density fill			
		s = np.nonzero(blocks[1] >= (blocks[2] * verticalthresh))
		blocks[2][s[0],s[1]] += density * (np.random.random(len(s[0])) < 0.5)

	return locs, blocks

def plotlocs(locs,blocks,blockwidth=100):
	blocksize = len(blocks[0])
	mesh = np.linspace(0,blocksize*blockwidth,blocksize+1)
	mxv,myv = np.meshgrid(mesh,mesh)
	plt.pcolormesh(mxv,myv,blocks[0])
	plt.scatter(locs[2],locs[1],s=10,c='r')
	plt.axes().set_aspect(1.0);
	plt.show()



if __name__ == "__main__":
	# locs,blocks = genlocs(n=20000,blockwidth=250,blocksize=100,affinity=0.0);
	# plotlocs(locs,blocks,blockwidth=250)
	genActivities(500,50,10);
