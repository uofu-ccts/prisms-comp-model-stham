import numpy as np;
import pandas as pd;
import time;

import multiprocessing as mp;


def funky(x):
	print(x);
	print(np.sum(bigarray))
	time.sleep(5);


def main():
	

l
	global bigarray;
	bigarray = np.random.randn(100000000)
	print("Bytes ", bigarray.nbytes);

	p = mp.Pool(processes=8)

	# df = pd.DataFrame(np.random.randn(100,10))

	# df = np.random.randn(100,10);
	g = 20
	a = np.random.randint(0,g,size=100)
	b = np.random.randint(0,g,size=100)
	c = np.random.randint(0,g,size=100)

	df= pd.DataFrame({'a':a,'b':b,'c':c});
	
	print(df)

	p.map(funky,df.groupby('a'),chunksize=2)

if __name__ == "__main__":
	main();
