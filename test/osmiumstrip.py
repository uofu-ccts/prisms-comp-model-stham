import os
import pandas as pd;

datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
osmium = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/osmosis/bin/osmosis"
utahlatest = datapath + "utah-latest.osm.pbf"
infile = " --read-pbf file=" + utahlatest

atrvals = pd.read_csv(datapath + "atrvals.csv")
delta = 0.02

for ind,fr in atrvals.iterrows():


	bbox = " --bounding-box top="+str(fr.lat+delta)+" left="+str(fr.long-delta)+" bottom="+str(fr.lat-delta)+" right="+str(fr.long+delta)
	out = " --write-pbf file="+ datapath + "utah-latest"+str(int(fr.atrnum))+".osm.pbf"
	os.system(osmium + infile + bbox + out)
	# break;
	# print(osmium + infile + bbox + out);