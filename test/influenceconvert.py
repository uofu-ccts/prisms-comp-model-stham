import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import datetime
import h5py;
import sqlite3
import multiprocessing as mp;
import mysql.connector


datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"

# day = int(sys.argv[1])

def reproc(day):
	infile = h5py.File(datapath + "Finfluence"+str(day)+".h5", 'r')
	rawmat = infile['population'+str(day)][:24,:3,:,:];
	xorigin = infile['population'+str(day)].attrs['xorigin']
	yorigin = infile['population'+str(day)].attrs['yorigin']
	grid = infile['population'+str(day)].attrs['grid']
		
	#swap because axes get jacked by hdf5
	rawmat = np.swapaxes(rawmat,2,3)
	xsize = rawmat[0,0].shape[0]
	ysize = rawmat[0,0].shape[1]
	
	records = np.zeros(24*xsize*ysize,dtype='uint16,uint16,float32,float32,uint32,uint32,uint32');
	#records = np.zeros(10,dtype='uint16,uint16,float32,float32,uint32,uint32,uint32');
	
	#print(xsize,ysize);
	
	c = 0;
	for t in range(24):
		for x in range(xsize):
			for y in range(ysize):
				if( np.sum( [rawmat[t,0,x,y],rawmat[t,1,x,y],rawmat[t,2,x,y]] ) > 0 ):
					records[c] = (day, t,x*grid + xorigin,y*grid + yorigin,rawmat[t,0,x,y],rawmat[t,1,x,y],rawmat[t,2,x,y])
					c += 1;
# 				if(c >= 10): break;
# 			if(c >= 10): break;
# 		if(c >= 10): break;
				
	#outframe.append(pd.DataFrame(records));
	return np.resize(records,c);



#outframe = pd.DataFrame(columns=['day','hour','x','y','pResidential','pWork','pTravel']);
#outmat = [];
print(datetime.datetime.now().time().isoformat());

p = mp.Pool(7);

vals = list(range(1,8));
vals = [(b,) for b in vals]

outmat = p.starmap(reproc,vals)

print(datetime.datetime.now().time().isoformat());

#print(outmat)

outmat = np.concatenate(outmat)

#print(outmat);

#outframe = pd.DataFrame(data = outmat, columns=['day','hour','x','y','pResidential','pWork','pTravel']);

#force datatypes
#keys = ['day','hour','pResidential','pWork','pTravel']
#for i in keys: outframe[i] = outframe[i].astype('uint32');

sqlite3.register_adapter(np.uint16, np.int)
sqlite3.register_adapter(np.uint32, np.int)
sqlite3.register_adapter(np.float32, np.long)

con = sqlite3.connect("/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/influence7sm.sq3");
#outframe.to_sql('influence',con,index=False);

c = con.cursor();
c.execute('''CREATE TABLE influence (day int, hour int, x real, y real, pResidential int, pWork int, pTravel int)''')
c.executemany('INSERT into influence VALUES (?,?,?,?,?,?,?)', outmat);

con.commit();
con.close();

#print(out);

# ds.attrs['xorigin'] = x * 500.0;
# ds.attrs['yorigin'] = y * 500.0;
# ds.attrs['day'] = day;
# ds.attrs['date']=datetime.datetime.now().isoformat() #place holder
# ds.attrs['grid']=500.0

