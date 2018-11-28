import pandas as pd;
import numpy as np;
import rtree;
import locations
import joblib
from enums import PTe,AWe,LWe;




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
		tripmethod = None

	wherecount = len(ptab[PTe.WHEREPROB][0])
	lengths = np.zeros(ptab[PTe.ACTCOUNT])
	lmaxv = np.zeros(ptab[PTe.ACTCOUNT])
	where = np.zeros(ptab[PTe.ACTCOUNT])
	for i in range(ptab[PTe.ACTCOUNT]):
		lwinpick = np.random.choice(ptab[PTe.LENCOUNT],p=ptab[PTe.LENACTJOINTPROB][i])
		lhistpick = np.random.choice(len(ptab[PTe.LHIST][lwinpick]),p=ptab[PTe.LHIST][lwinpick])
		low = ptab[PTe.LBINS][lwinpick][lhistpick]
		high = ptab[PTe.LBINS][lwinpick][lhistpick + 1]
		lengths[i] = np.random.rand() * (high-low) + low
		lmaxv[i] = ptab[PTe.LENWINS][lwinpick][LWe.LMAX.value]
		where[i] = np.random.choice(wherecount,p=ptab[PTe.WHEREPROB][i])

	precorder = precsort(ptab[PTe.ORDERPROB])
	newtraj = np.copy(ptab[PTe.ACTWINS])
	# print(newtraj[0])
	newtraj = np.append(newtraj,lengths.reshape(ptab[PTe.ACTCOUNT],1),axis=1)

	newtraj = np.append(newtraj,precorder.reshape(ptab[PTe.ACTCOUNT],1),axis=1)
	newtraj = np.append(newtraj,lmaxv.reshape(ptab[PTe.ACTCOUNT],1),axis=1)
	newtraj = np.append(newtraj,where.reshape(ptab[PTe.ACTCOUNT],1),axis=1)
	sorttraj = np.lexsort(newtraj.T[[AWe.WMAX.value,AWe.LENS.value,AWe.WMIN.value,AWe.WAVG.value,AWe.PREC.value]])
	# print(sorttraj)
	print(newtraj)
	newtraj = newtraj[sorttraj].T
	print(newtraj)

	#pickwins up to 3 times
	actpicks = np.full(ptab[PTe.ACTCOUNT],False)
	validwin = np.full(ptab[PTe.ACTCOUNT],True)
	actprob = newtraj[AWe.ACTPROB.value]
	for i in range(3):
		print(actprob)
		actpicks = actpicks | ((np.random.rand(ptab[PTe.ACTCOUNT]) < actprob ) & validwin)
		ends = np.cumsum(newtraj[AWe.LENS.value]*actpicks)
		starts = (ends*actpicks) - newtraj[AWe.LENS.value]
		
		validwin = (starts <= newtraj[AWe.WMAX.value]) & (starts >= newtraj[AWe.WMIN.value])

		if(np.sum(newtraj[AWe.LMAX.value]*actpicks)) > 1440:
			break;
		else:
			#mass adjustment
			actprob += 0.2
	print(np.sum(newtraj[AWe.LMAX.value]*actpicks),actpicks)
	newtraj = newtraj.T[actpicks]
		

	#setlocs


	#get x/y
	xcoord = np.zeros(len(newtraj))
	ycoord = np.zeros(len(newtraj))
	#get trips

	#smear


	return newtraj;



def runit():

	#FIXME: remove load or add test path
	ptab = joblib.load("windowtest.gz")

	print(buildtraj(0,0,ptab));


if __name__=="__main__":
	runit();