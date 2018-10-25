import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import h5py
import sqlite3
import pyproj
import matplotlib.cm as cm
import datetime
import sys;
import time;


blocksize = 10000
gridsize = 250.0
stepsize=60
regmaxlat = 40.82
regminlat = 40.46
regminlong = -112.20
regmaxlong = -111.77

outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');
def latlongtrans(x):
	x.locx,x.locy=pyproj.transform(inproj,outproj,x.locx,x.locy);
	return x

def reversetrans(x):
	x.locx,x.locy=pyproj.transform(outproj,inproj,x.locx,x.locy);
	return x

def latlongtransraw(locx,locy):
	locx,locy=pyproj.transform(inproj,outproj,locx,locy);
	return locx,locy

def reversetransraw(locx,locy):
	locx,locy=pyproj.transform(outproj,inproj,locx,locy);
	return locx,locy

mindim = reversetransraw(regminlong,regminlat);
maxdim = reversetransraw(regmaxlong,regmaxlat);

long0,lat0 = reversetransraw(-111.8723,40.7341)
print(long0,lat0)

# def dist(lat,long):
# 	long,lat = reversetransraw(long,lat)
# 	dlong =  long0 - long; dlat = lat0 - lat
# 	return np.sqrt( dlat * dlat + dlong * dlong);

def bresenham(x0,y0,x1,y1):
	dx = x1-x0
	dy = y1-y0
	sx = np.sign(dx)
	sy = np.sign(dy)
	dx = np.abs(dx);
	dy = np.abs(dy)
	err = dx - dy

	out = [(x0,y0)]
	x = x0;
	y = y0;

	while(x != x1 or y != y1):
		e2 = err << 1
		if(e2 > -dy):
			err -= dy
			x += sx
		if(e2 < dx):
			err += dx
			y += sy
		out += [(x,y)]
	return out;

def dist(x0,y0,x1,y1):
	dx = x1-x0; dy = y1-y0;
	return np.sqrt(dx*dx+dy*dy)

def splitline(x0,y0,x1,y1,grid=1.0):
	if(x0 == x1 and y0 == y1):
		return [[x0,y0,1.0]]

	flipflag = False;
	if(x0 == x1):
		flipflag = True
		x0,y0,x1,y1 = y0,x0,y1,x1

	points = [[x0,y0],[x1,y1]];
	# print(points)
	dx = x1-x0
	dy = y1-y0

	minx = np.minimum(x0,x1)
	miny = np.minimum(y0,y1)
	maxx = np.maximum(x0,x1)
	maxy = np.maximum(y0,y1)
	
	offx = grid - np.fmod(minx,grid)
	offy = grid - np.fmod(miny,grid)

	m = dy/dx
	b = y0 - m*x0
	
	for x in np.arange(minx+offx,maxx,grid):
		points += [[x,m*x+b ]]

	for y in np.arange(miny+offy,maxy,grid):
		points += [[ (y - b)/m,y]]

	points = np.array(points)
	plist = np.lexsort((points.T[0],points.T[1]))
	points = points[plist]

	if flipflag:
		points = points.T
		flip = np.array([points[1],points[0]])
		points = flip.T

	disc = []
	tdist = dist(x0,y0,x1,y1)
	for i in range(1,len(points)):
		d = dist(points[i][0],points[i][1],points[i-1][0],points[i-1][1])
		x = np.floor((points[i-1][0] +  points[i][0]) / 2)
		y = np.floor((points[i-1][1] +  points[i][1]) / 2)
		disc += [[int(x),int(y),d/tdist]]
	return disc;

def speedlimit(x0,y0,x1,y1,t,limit):
	if not (t > 0): return True;
	dx,dy = x1-x0,y1-y0
	d = np.sqrt(dx*dx + dy*dy)
	if(d/t > limit): return True;
	return False;

def processtraj(gframe,mats):

	steps = 1440 // stepsize;

	exp = np.zeros(steps);

	ind,frame = gframe;

	winmin = np.arange(0,1440,stepsize)
	winmax = winmin + stepsize

	xd,yd = mats[0].shape


	# x,y = 0,0
	# if(frame.iloc[0].long <= regmaxlong and frame.iloc[0].long >= regminlong and \
	# frame.iloc[0].lat <= regmaxlat and frame.iloc[0].lat >= regminlat):
	# 	locx,locy = reversetransraw(frame.iloc[0].long,frame.iloc[0].lat);
	# 	x,y = getgrid(locx,locy,gridsize,mindim);

	# lastact = [frame.iloc[0].actcode,frame.iloc[0].actcode,x,y]

	frame[['locx','locy']] = frame[['long','lat']]
	frame[['locx','locy']] = frame[['locx','locy']].apply(reversetrans,axis=1)
	frame['x'] = frame['locx'].apply(lambda x: (x - mindim[0])/gridsize)
	frame['y'] = frame['locy'].apply(lambda x: (x - mindim[1])/gridsize)
	frame['nx'] = frame['x'].shift(-1)
	frame['ny'] = frame['y'].shift(-1)

		# print(d);
	for ind,f in frame.iterrows():
		end = min(f.start+f.length,1439)
		#if( f.start < winmax and end > winmin):
		slotst = int(np.floor(f.start/stepsize))
		sloten = int(np.floor(end/stepsize))+1
		# print(f.actcode,f.start,slotst,end,sloten)
		for slot in range(int(slotst),int(sloten)):
			# winmin = slot*stepsize; winmax = winmin + stepsize;
			if( (f.x >= 0.0)& (f.y >= 0.0) & (f.x < xd) & (f.y < yd) ):
				weight = min(end, winmax[slot]) - max(f.start,winmin[slot])
				if(np.isnan(f.nx) & np.isnan(f.ny)):
					exp[slot] += weight * mats[slot][int(np.floor(f.x)),int(np.floor(f.y))];
				# locx,locy = reversetransraw(f.long,f.lat);
				# x,y = getgrid(locx,locy,gridsize,mindim);
				# if(x >= 0 and x < xd and y >=0 and y < yd):
				
				# print(weight)
				elif((f.nx < xd) & (f.ny < yd)):
					if(f.actcode >= 180000 and f.actcode <=189999):
						# if(lastact[0] == f.actcode and lastact[1] == f.agentnum):
						# 	#speed limit - we assume max speed of 50 m/s 
						# 	if not(speedlimit(x,y,lastact[2],lastact[3],weight, 50.0)):
						# 		# mats[slot][2][x,y] += weight;
						# 	# else:
						line = splitline(f.x,f.y,f.nx,f.ny)
						for t in range(len(line)):
							exp[slot] += (weight * line[t][2]) * mats[slot][int(line[t][0]),int(line[t][1])];

					else: 
						exp[slot] += weight * mats[slot][int(np.floor(f.x)),int(np.floor(f.y))];

				# lastact = [f.actcode,f.agentnum,x,y]



	return exp / 60.0;


def getseqs():
	blocks = (490351031001017,490351034003013,490351032001002,490351033003005,490351031001018,490351032001000,490351034003016,490351031001013,490351031001015,490351034003012,490351031001012,490351031001010,490351031001014,490351031001011,490351032001003,490351032001004,490351032001005,490351032001006,490351034003009,490351033003003,490351033003004,490351033003002,490351031001016,490351032001001)

	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	runpath = "/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run05/"

	datacon = sqlite3.connect(datapath + "indvs2.sq3")
	indvs = pd.read_sql_query("select * from indvs where block in " + str(blocks), datacon)
	datacon.close()

	indvlabels = pd.read_csv(datapath + "indvlabels.csv")


	agentnums = tuple(indvs["id"].values)

	# indvlabels = indvlabels[indvlabels["id"].isin(agentnums)];


	# caseset = list(set(indvlabels["casetype"].values))
	# print(caseset)
	# clindex = { b:i for i,b in enumerate(caseset) }
	# print(clindex)
	# x = np.linspace(0.0,1.0,len(caseset))
	# colors = [ cm.jet(b) for b in x ] 


	# print(indvlabels);
	# print(agentnums)
	# exit()
	runcon = sqlite3.connect(runpath + "Ftraj4-2018-09-13_15-55-21-ForkPoolWorker-10.merge.sqlite3")

	seqs = pd.read_sql_query("select * from acttraj where agentnum in " + str(agentnums), runcon)
	runcon.close();
	return seqs;



def getpm25():
	
	pmpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/horel/"
	index = "009"
	prefix = "pm_w_test_hr_"
	mats = []
	for i in range(0,24):
		fname = pmpath + index + "/" + prefix + index + "_" + str(i).zfill(2) + ".hdf"
		file = h5py.File(fname,'r');
		mats += [file["/values"][:,:]]
		file.close()
	return mats;




def runit():

	mats = getpm25();
	# print(mats);
	seqs = getseqs();
	gseqs = seqs.groupby("agentnum");
	casecount = len(gseqs);
	print(casecount)
	exp = np.zeros((casecount,24))
	c = 0;
	for gf in gseqs:
		if(c%10==0): print(c,end=' ',flush=True);
		exp[c] = processtraj(gf,mats)
		# plt.plot(exp[c],linewidth=0.5,alpha=0.5)
		c += 1;
	# plt.show();
	runpath = "/uufs/chpc.utah.edu/common/home/u0403692/bmi-group1/prism/run05/"
	# outpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
	file = h5py.File(runpath + "pm25traj.h5",'w')
	ds = file.create_dataset("/exppm25",data=exp,fillvalue=0.,compression='gzip',compression_opts=9)
	file.close();
	


if __name__ == "__main__":
	runit();



# seqs["end"] = seqs["start"]+seqs["length"]


# runcon.close()

# seqs['d'] = seqs[["long","lat"]].apply(lambda x: dist(x.lat,x.long),axis=1)

# gseq = seqs.groupby("agentnum")

# lats,longs = [-111.8723,-111.8873,-111.8436],[40.7341,40.7769,40.7638]

# # plt.scatter(lats,longs,s=100)
# # for i,gf in gseq:
# #     plt.plot(gf["long"],gf["lat"],linewidth=0.1)
# # plt.axes().set_aspect(1.0)
# # plt.show()

# stats = np.zeros((len(indvlabels),2));

# # for c in caseset:
# count = 0;

# for i,gf in gseq:
# 	# print(gf.iloc[0].loc['agentnum'])
# 	 
# 	weight = np.mean(gf["length"]/1440.0 * gf["d"])
	
# 	stats[count] = np.array([case,weight]);
# 	# if case == c:
# 	count += 1;
# 	# plt.plot(gf["end"],gf["d"],linewidth=0.5,alpha=0.5, color=colors[clindex[case]])

# print(stats)

# # plt.title(str(c)+", count:"+str(count))
# # plt.show()
# # plt.scatter(stats.T[0],stats.T[1],s=10);
# H,xe,ye = np.histogram2d(stats.T[0],stats.T[1],bins=[len(caseset),100])
# plt.xticks(xe,[str(b) for b in xe]);
# # X,Y = np.meshgrid(xe[:-1],ye[:-1])
# plt.pcolormesh(H);
# plt.show()




#CODE DUMP
# for ind,i in enumerate(caseset):
#     st = modms[indvlabels["casetype"].values == i]
#     plt.errorbar(ind,np.mean(st)/np.mean(modms),yerr=np.std(st)/np.std(modms),fmt='o',c='r')
#     st = ms[indvlabels["casetype"].values == i]
#     plt.errorbar(ind+len(blocks)+len(caseset),np.mean(st)/np.mean(ms),yerr=np.std(st)/np.std(ms),fmt='^',c='r')
# for ind,i in enumerate(blocks):
#     st = modms[indvs["block"].values == i]
#     plt.errorbar(ind+len(caseset),np.mean(st)/np.mean(modms),yerr=np.std(st)/np.std(modms),fmt='o',c='b')
#     st = ms[indvs["block"].values == i]
#     plt.errorbar(ind+0.5+len(blocks)+len(caseset)*2,np.mean(st)/np.mean(ms),yerr=np.std(st)/np.std(ms),fmt='^',c='b')
# plt.show()


# for ind,i in enumerate(caseset):
#     st = modms[indvlabels["casetype"].values == i]
#     plt.errorbar(ind,1,yerr=np.std(st)/np.std(modms),fmt='o',c='r')
#     st = ms[indvlabels["casetype"].values == i]
#     plt.errorbar(ind+len(blocks)+len(caseset),1,yerr=np.std(st)/np.std(ms),fmt='^',c='r')
# for ind,i in enumerate(blocks):
#     st = modms[indvs["block"].values == i]
#     plt.errorbar(ind+len(caseset),1,yerr=np.std(st)/np.std(modms),fmt='o',c='b')
#     st = ms[indvs["block"].values == i]
#     plt.errorbar(ind+0.5+len(blocks)+len(caseset)*2,1,yerr=np.std(st)/np.std(ms),fmt='^',c='b')
# plt.show()

# for ind,i in enumerate(caseset):
#     # st = modms[indvlabels["casetype"].values == i]
#     # plt.errorbar(ind,1,yerr=np.std(st)/np.std(modms),fmt='o',c='r')
#     st = ms[indvlabels["casetype"].values == i]
# 	plt.errorbar(ind,np.mean(st),yerr=np.std(st),fmt='^',c='r')
# for ind,i in enumerate(blocks):
#     # st = modms[indvs["block"].values == i]
#     # plt.errorbar(ind+len(caseset),1,yerr=np.std(st)/np.std(modms),fmt='o',c='b')
#     st = ms[indvs["block"].values == i]
# 	plt.errorbar(ind+len(caseset),np.mean(st),yerr=np.std(st),fmt='^',c='b')
# plt.show()


# file = h5py.File(grouppath + "hawthexptraj-0.h5",'r')
# modmat = file["/exptraj"][:]
# modmat = np.reshape(modmat,(1858,96))
# file.close()
# modms = np.sum(modmat,axis=1)
# smodms = modms[np.argsort(indvlabels["casetype"].values)]
# plt.scatter(np.arange(1858),smodms/np.mean(smodms),s=10,c=np.sort(indvs["blockid"].values),cmap="jet",marker='^',alpha=1.0)
# #plt.scatter(np.arange(1858),ms/np.mean(ms),s=10,c=indvs["blockid"].values,cmap="jet",marker='o',alpha=0.5)
# plt.ylim(0.0,4.0)
# plt.show()


# file = h5py.File(grouppath + "hawthexptraj-0.h5",'r')
# modmat = file["/exptraj"][:]
# modmat = np.reshape(modmat,(1858,96))
# file.close()
# modms = np.mean(modmat,axis=1)
# modms = modms[np.argsort(indvs["blockid"].values)]
# smodms = modms[np.argsort(indvlabels["casetype"].values)]
# ax = plt.subplot(211)
# ax.scatter(np.arange(1858),modms,s=5,c=np.sort(indvs["blockid"].values),cmap="viridis",marker='^',alpha=0.75)
# #ax.scatter(np.arange(1858),ms,s=5,c=np.sort(indvs["blockid"].values),cmap="viridis",marker='o',alpha=0.75)
# for i in np.cumsum(indvs["blockid"].value_counts(sort=False).values):
#     ax.errorbar(i,1.0,yerr=1000000.0,c='k',linewidth=0.5)
# #ax.set_ylim(0.98,1.10)
# #ax.set_ylim(620,690)
# #ax.set_ylim(26,29)
# ax.set_ylim(25000,40000)
# ax.set_xlim(-10,1900)
# ax = plt.subplot(212)
# ax.scatter(np.arange(1858),smodms,s=5,c=indvs["blockid"].values[np.argsort(indvlabels["casetype"].values)],cmap="viridis",marker='^',alpha=0.75)
# #ax.scatter(np.arange(1858),sms,s=5,c=indvs["blockid"].values[np.argsort(indvlabels["casetype"].values)],cmap="viridis",marker='o',alpha=0.75)
# for i in np.cumsum(indvlabels["casetype"].value_counts(sort=False).values):
#     ax.errorbar(i,1.0,yerr=1000000.0,c='k',linewidth=0.5)
# #ax.set_ylim(0.98,1.10)
# #ax.set_ylim(620,690)
# #ax.set_ylim(26,29)
# ax.set_ylim(25000,40000)
# ax.set_xlim(-10,1900)
# plt.show()


# file = h5py.File(grouppath + "hawthexptraj-0.h5",'r')
# modmat = file["/exptraj"][:]
# modmat = np.reshape(modmat,(1858,96))
# file.close()
# modms = np.mean(modmat,axis=1)
# modms = modms[np.argsort(indvs["blockid"].values)]
# smodms = modms[np.argsort(indvlabels["casetype"].values)]
# ax = plt.subplot(211)
# #ax.scatter(np.arange(1858),modms,s=5,c=np.sort(indvs["blockid"].values),cmap="viridis",marker='^',alpha=0.75)
# ax.scatter(np.arange(1858),ms,s=3,c=np.sort(indvs["blockid"].values),cmap="viridis",marker='o',alpha=0.75,linewidth=0.5)
# for i in np.cumsum(indvs["blockid"].value_counts(sort=False).values):
#     ax.errorbar(i,1.0,yerr=1000000.0,c='k',linewidth=0)
# #ax.set_ylim(0.98,1.10)
# #ax.set_ylim(620,690)
# ax.set_ylim(25,29)
# #ax.set_ylim(0000,40000)
# ax.set_xlim(-10,1900)
# ax.set_xticks([],[])
# ax = plt.subplot(212)
# #ax.scatter(np.arange(1858),smodms,s=5,c=indvs["blockid"].values[np.argsort(indvlabels["casetype"].values)],cmap="viridis",marker='^',alpha=0.75)
# ax.scatter(np.arange(1858),sms,s=3,c=indvs["blockid"].values[np.argsort(indvlabels["casetype"].values)],cmap="viridis",marker='o',alpha=0.75,linewidth=0.5)
# for i in np.cumsum(indvlabels["casetype"].value_counts(sort=False).values):
#     ax.errorbar(i,1.0,yerr=1000000.0,c='k',linewidth=0)
# #ax.set_ylim(0.98,1.10)
# #ax.set_ylim(620,690)
# ax.set_ylim(25,29)
# #ax.set_ylim(0000,40000)
# ax.set_xlim(-10,1900)
# ax.set_xticks([],[])
# plt.tight_layout()
# F = plt.gcf()
# F.set_size_inches(7,3.5)
# F.savefig("pm25expcaseblock.eps")
# #plt.show()


