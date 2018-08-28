import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;



def genActivities(n):
	pass;


	

def genDemographics(n):
	pass;


def affexp(x,aff):
	k = np.power(2,aff)
	return np.power(2,x*k)/np.power(2,k)

def rloc(x,y,r):
	s = np.random.random() * 2 * np.pi
	cs = np.cos(s); sn = np.sin(s);
	div = np.max(l
	return x*r + xs*r, y*r + ys*r

def genlocs(n = 1000, blocksize = 10,blockwidth = 100, majorroad = 10, density = 100, verticalprob = 0.10, mode='single', weights = None, affinity = 2.0, seed = None):
	""" 
	Creates an artificial list of locations
	n - total number of locations to populate
	blockwidth - size of a block, meters
	majorroad - number of blocks before a major roadway is built
	density - maximum number of locations per block before vertical scaling test occurs
	verticalprob - probability of additional vertical units being added at capacity
	mode - single or inclusive zoning method
	weights - relative weights of location types
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

	types = ['Residential','Business','Commercial','Industrial','Institutional','Open','Roadway']

	# radius = n / density
	# blocksize = int(np.floor(2*radius))+1
	blocksize += 1

	blocks = np.zeros((3,blocksize,blocksize)) #type,count,max

	if(weights == None):
		weights = [0.4,0.15,0.15,0.15,0.05,0.10]

	locs = np.zeros((3,n)) #type, x, y

	if mode == 'single':
		blocks[0] = -1
		blocks[2] = density
		for i in range(0,blocksize,10):
			blocks[0][i] = 6
			blocks[0].T[i] = 6			

		
		locs[0] = np.random.choice(6,p=weights,size=n)
		locs[0][0:7] = np.arange(7);
		for i in range(n):
			s = np.nonzero(blocks[0] == locs[0][i])
			ms = len(s)
			s = np.append(s,np.nonzero(blocks[0] == -1),axis=1)
			p = affexp(blocks[2][s[0],s[1]] - blocks[1][s[0],s[1]],affinity)
			p[ms:] *= 0.03
			pick = np.random.choice(len(p),p=p/np.sum(p))
			locs[1][i],locs[2][i] = rloc(s[0][pick],s[1][pick],blockwidth)
			blocks[1][s[0][pick],s[1][pick]] += 1
			if(blocks[0][s[0][pick],s[1][pick]] == -1):
				blocks[0][s[0][pick],s[1][pick]] = locs[0][i]

			s = np.nonzero(blocks[1] == blocks[2])
			blocks[2][s[0],s[1]] += density * (np.random.random(len(s[0])) < verticalprob)

		print(blocks)
		
		mesh = np.linspace(0,blocksize*blockwidth,blocksize+1)
		mxv,myv = np.meshgrid(mesh,mesh)
		plt.pcolormesh(mxv,myv,blocks[0])
		plt.scatter(locs[1],locs[2],s=10,c='r')
		plt.axes().set_aspect(1.0);
		plt.show()		

	elif mode == 'inclusive':
		pass;

	else:
		print("Invalid mode given for genlocs!");
		return None


def genAgents(n):
	pass;

if __name__ == "__main__":
	genlocs();