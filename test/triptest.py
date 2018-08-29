import pycurl;
from io import BytesIO;
import json;
import numpy as np;
import pyproj;
import pandas as pd;
import matplotlib.pyplot as plt;
#import geojson;


outproj = pyproj.Proj(init='epsg:4326');
inproj = pyproj.Proj(init='epsg:26912');
def latlongtrans(x):
	x.locx,x.locy=pyproj.transform(inproj,outproj,x.locx,x.locy);
	return x

def reversetrans(x):
	x.locx,x.locy=pyproj.transform(outproj,inproj,x.locx,x.locy);
	return x

baseopts= "?alternatives=5&overview=full&geometries=geojson&annotations=duration"

def gettrip(baseurl,lonx1,laty1,lonx2,laty2):
	curlobj = pycurl.Curl()
	data = BytesIO()
	baseurl = "http://0.0.0.0:30606/route/v1/driving/"
	baseopts= "?alternatives=3&overview=full&geometries=geojson&annotations=duration"

	#curlobj.setopt(curlobj.URL, "http://0.0.0.0:30606/route/v1/driving/-111.850805,40.767031;-111.8808,40.777031?alternatives=5")
	curlobj.setopt(curlobj.URL, baseurl +str(lonx1)+","+str(laty1)+";"+str(lonx2)+","+str(laty2)+baseopts);
	curlobj.setopt(curlobj.WRITEFUNCTION, data.write)
	curlobj.perform()
	# print(data.getvalue());
	dc = json.loads(data.getvalue())
	# print(dc);
	if('routes' in dc):
		rcount = len(dc['routes']);
		pick = np.random.randint(0,rcount);
		# print(len(dc['routes'][pick]['geometry']['coordinates']), len(dc['routes'][pick]['legs'][0]['annotation']['distance']))
		line = np.array(dc['routes'][pick]['geometry']['coordinates']).T
		duration = np.array(dc['routes'][pick]['legs'][0]['annotation']['duration'] + [0.0])
		dur = np.sum(duration)
		
		df = pd.DataFrame({"locx":line[0],"locy":line[1],"length":duration});
		df['length'] = (df['length']/60.0)

		# df[['locx','locy']] = df[['locx','locy']].apply(reversetrans, axis=1);
		df['triporder'] = np.arange(0,len(df));
		return df,dur;
	else:
		print("ERROR DETECTED IN ROUTE: dumping json");
		print(str(data.getvalue()))
		return None,None;

def getwp(baseurl,lonx,laty):
	curlobj = pycurl.Curl()
	data = BytesIO()
	baseurl = "http://0.0.0.0:30606/nearest/v1/driving/"
	baseopts= "?number=1"

	#curlobj.setopt(curlobj.URL, "http://0.0.0.0:30606/route/v1/driving/-111.850805,40.767031;-111.8808,40.777031?alternatives=5")
	curlobj.setopt(curlobj.URL, baseurl +str(lonx)+","+str(laty)+baseopts);
	curlobj.setopt(curlobj.WRITEFUNCTION, data.write)
	curlobj.perform()
	# print(data.getvalue());
	dc = json.loads(data.getvalue())
	print(dc);
	if('waypoints' in dc):
		# rcount = len(dc['waypoints']);
		# pick = np.random.randint(0,rcount);
		# print(len(dc['routes'][pick]['geometry']['coordinates']), len(dc['routes'][pick]['legs'][0]['annotation']['distance']))
		# line = np.array(dc['routes'][pick]['geometry']['coordinates']).T
		# duration = np.array(dc['routes'][pick]['legs'][0]['annotation']['duration'] + [0.0])
		# dur = np.sum(duration)
		print(dc['waypoints'][0]['location'])
		line = np.array(dc['waypoints'][0]['location']).T
		# df = pd.DataFrame({"locx":line[0],"locy":line[1]});
		# df['length'] = (df['length']/60.0)

		# df[['locx','locy']] = df[['locx','locy']].apply(reversetrans, axis=1);
		# df['triporder'] = np.arange(0,len(df));
		return line;
	else:
		print("ERROR DETECTED IN ROUTE: dumping json");
		print(str(data.getvalue()))
		return None;



datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/"
baseurl = "http://0.0.0.0:30606/route/v1/driving/"

# tr = gettrip(baseurl,-111.850805,40.767031,-111.8808,40.777031);

atrvals = pd.read_csv(datapath + "atrvals.csv")

for ind,fr in atrvals.iterrows():

	# tr = getwp(baseurl,fr.long,fr.lat)
	tr,dur = gettrip(baseurl,-111.89131,40.52539,-111.89096,40.52549)
	# tr,dur = gettrip(baseurl,fr.long,fr.lat-0.0002,fr.long+0.0002,fr.lat)
	if(type(tr) != type(None)):
		print(tr)
		plt.scatter(tr.locx.values,tr.locy.values,s=10)
		# plt.scatter(tr[0],tr[1],s=10)
		plt.scatter(-111.89152,40.51650,s=20)
		# plt.scatter(fr.long,fr.lat,s=20)
	break
plt.show()
# print(tr);