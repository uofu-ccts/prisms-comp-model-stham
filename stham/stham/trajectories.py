import pandas as pd;
import numpy as np;
import rtree;
import locations
from enum import Enum


class PTe(Enum):
	ACTCOUNT = 0
	LENCOUNT = 1
	LENACTJOINTPROB = 2
	ACTWINS = 3
	LENWINS = 4
	LHIST = 5
	LBINS = 6
	LHISTLEN = 7
	ORDERPROB = 8
	WHEREPROB = 9


class AWe(Enum):
	ACTCODE = 0
	ACTPROB = 1
	WMIN = 2
	WMAX = 3
	WAVG = 4
	LENS = 5 #only in newtraj
	PREC = 6 #only in newtraj
	LMAX = 7 #only in newtraj
	LOC = 8 #only in newtraj

class LWe(Enum):
	LMIN = 0
	LMAX = 1




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

def buildtraj(agent, locs, ptab, tripmethod=None):
	
	
	if(tripmethod == None):
		tripmethod = 

	wherecount = len(PTe.WHEREPROB[0])
	lengths = np.zeros(ptab[PTe.ACTCOUNT])
	lmaxv = np.zeros(ptab[PTe.ACTCOUNT])
	where = np.zeros(ptab[PTe.ACTCOUNT])
	for i in range(ptab[PTe.ACTCOUNT]):
		lwinpick = np.random.choice(ptab[PTe.LENCOUNT],p=ptab[PTe.LENACTJOINTPROB][i])
		lhistpick = np.random.choice(ptab[PTe.LHISTLEN],p=ptab[PTe.LHIST][lwinpick])
		low = ptab[PTe.LBINS][lwinpick][lhistpick]
		high = ptab[PTe.LBINS][lwinpick][lhistpick + 1]
		lengths[i] = np.random.rand() * (high-low) + low
		lmaxv[i] = ptab[PTe.LENWINS][lwinpick][LWe.LMAX]
		where[i] = np.random.choice(wherecount,p=ptab[Pte.WHEREPROB][i])

	precorder = precsort(ptab[PTe.ORDERPROB])
	newtraj = np.copy(ptab[PTe.ACTWINS])
	newtraj = np.append(newtraj,lengths,axis=0)
	newtraj = np.append(newtraj,precorder,axis=0)
	newtraj = np.append(newtraj,lmaxv,axis=0)
	newtraj = np.append(newtraj,where,axis=0)
	sorttraj = np.lexsort(newtraj[[AWe.WMAX,AWe.LENS,AWe.WMIN,AWe.WAVG,AWe.PREC]])
	newtraj = newtraj.T[sorttraj].T

	#pickwins up to 3 times
	actpicks = np.full(ptab[PTe.ACTCOUNT],False)
	validwin = np.full(ptab[PTe.ACTCOUNT],True)
	for i in range(3):
		actpicks = actpicks | ((np.random.rand(ptab[PTe.ACTCOUNT]) < newtraj[AWe.ACTPROB]) & validwin)
		ends = np.cumsum(newtraj[AWe.LENS]*actpicks)
		starts = (ends*actpicks) - newtraj[AWe.LENS]
		validwin = True if (start <= newtraj[AWe.WMAX] and start >= newtraj[AWe.WMIN]) else False
		if(newtraj[AWe.LMAX]*actpicks).sum() > 1440:
			break;

	newtraj = newtraj.T[actpicks].T
		

	#setlocs


	#get x/y
	xcoord = np.zeros(len(newtraj))
	ycoord = np.zeros(len(newtraj))
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