from mpi4py import MPI;
import pandas as pd;
import numpy as np;



comm = MPI.COMM_WORLD
rank = comm.Get_rank();

if rank == 0:
	data=pd.DataFrame(np.arange(1000,dtype='i'))
	data['t']=1.0
	# print(data)
	comm.send(data,dest=1,tag=10)
elif rank == 1:
	# buff = pd.DataFrame(np.empty((2,1000)));
	# buff = pd.DataFrame();
	# buff.columns = ['0','t']
	buff = comm.recv(source=0,tag=10)
	print(buff)

print(rank);