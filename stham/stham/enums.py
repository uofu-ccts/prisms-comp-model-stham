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