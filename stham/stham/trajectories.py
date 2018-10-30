import pandas as pd;
import numpy as np;
import rtree;


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

#enum for ACTWINS
ACTCODE = 0
ACTPROB = 1
WMIN = 2
WMAX = 3
WAVG = 4
LENS = 5 #only in newtraj
PREC = 6 #only in newtraj
LMAX = 7 #only in newtraj





#agent - pd,series, locs - dict of rtree, ptab - dict of probabilities

def precsort(precede):
	actlen = len(precede);
	
	pmat = np.random.rand(actlen,actlen);
	omat = np.zeros((actlen,actlen));

	for i in range(actlen):
		for j in range(i+1,actlen):
			result = pmat[i,j] < precede[i,j]
			omat[i,j] = (1.0 if result else 0.0)
			omat[j,i] = (0.0 if result else 1.0)
	
	return np.sum(omat, axis=0);



def buildtraj(agent, locs, ptab):
	
	lengths = np.zeros(ptab[ACTCOUNT])
	lmaxv = np.zeros(ptab[ACTCOUNT])
	for i in range(ptab[ACTCOUNT]):
		lwinpick = np.random.choice(ptab[LENCOUNT],p=ptab[LENACTJOINTPROB][i])
		lhistpick = np.random.choice(ptab[LHISTLEN],p=ptab[LHIST][lwinpick]
		low = ptab[LBINS][lwinpick][lhistpick]
		high = ptab[LBINS][lwinpick][lhistpick + 1]
		lengths[i] = np.random.rand() * (high-low) + low
		lmaxv[i] = ptab[LENWINS][lwinpick][LMAX]

	precorder = precsort(ptab[ORDERPROB])
	newtraj = np.copy(ptab[ACTWINS])
	newtraj = np.append(newtraj,lengths,axis=0)
	newtraj = np.append(newtraj,precorder,axis=0)
	newtraj = np.append(newtraj,lmaxv,axis=0)
	sorttraj = np.lexsort(newtraj[[WMAX,LENS,WMIN,WAVG,PREC]])
	newtraj = newtraj.T[sorttraj].T

	#pickwins up to 3 times
	actpicks = np.full(ptab[ACTCOUNT],False)
	validwin = np.full(ptab[ACTCOUNT],True)
	for i in range(3):
		actpicks = actpicks | ((np.random.rand(ptab[ACTCOUNT]) < newtraj[ACTPROB]) & validwin
		ends = np.cumsum(newtraj[LENS]*actpicks)
		starts = (ends*actpicks) - newtraj[LENS]
		validwin = True if start <= newtraj[WMAX] and start >=newtraj[WMIN] else False
		if(newtraj[LMAX]*actpicks).sum() > 1440):
			break;

	newtraj = newtraj.T[actpicks].T
		

	#setlocs

	#get x/y

	#get trips

	#smear


	return newtraj;



def runit():


	# for i in range(10):
		

	# agent = 0;
	# loc = 0;
	# ptab = {ACTCOUNT:10,LENCOUNT:,LENACTJOINTPROB:,ACTWINS:,:LENWINS:,LHIST:,LBINS:,LHISTLEN:,ORDERPROB:}


	pass;


if __name__=="__main__":
	runit();