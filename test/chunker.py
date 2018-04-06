import multiprocessing as mp;
import numpy as np;
import matplotlib.pyplot as plt;
import sys;
import time;


simstep = 50000
damp = 0.95
sims = 1000000
simchunk = 10000 #must be divisible by threads
threads = 4 

def chunker(size, chunksize, conn):
	print("chunked!")
	sys.stdout.flush();
	for i in range(0, size, chunksize):
		a = np.random.randn(chunksize);

		conn.put(a);
		time.sleep(20);
	conn.put([0,0,0]);

def worker(data):
	for i in range(simstep):
		data *= -damp

	return data;
	

def workerboss(workers, conn):
	p = mp.Pool(threads);
	print("bossy")
	result = [];
	
	while(True):
		a = conn.get();
		if(type(a) == list):
			break;
		else:
			split = np.split(a,workers)

			out = p.map(worker,split);
			result += [out];
	


	return result



def control():

	conn = mp.Queue();
	ch = mp.Process(target=chunker, args=(sims,simchunk,conn));
	ch.start();
	result = workerboss(threads,conn)
	ch.join();
	print(result);

if __name__ == '__main__':
	control();