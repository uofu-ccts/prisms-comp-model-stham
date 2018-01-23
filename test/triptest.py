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
	c = pycurl.Curl()
	data = BytesIO()



	#curlobj.setopt(curlobj.URL, "http://0.0.0.0:30606/route/v1/driving/-111.850805,40.767031;-111.8808,40.777031?alternatives=5")
	curlobj.setopt(curlobj.URL, baseurl +str(lonx1)+","+str(laty1)+";"+str(lonx2)+","+str(laty2)+baseopts);
	curlobj.setopt(curlobj.WRITEFUNCTION, data.write)
	curlobj.perform()
	# print(data.getvalue());
	dc = json.loads(data.getvalue())
	rcount = len(dc['routes']);
	pick = np.random.randint(0,rcount);
	print(pick)
	# print(len(dc['routes'][pick]['geometry']['coordinates']), len(dc['routes'][pick]['legs'][0]['annotation']['distance']))
	line = np.array(dc['routes'][pick]['geometry']['coordinates']).T
	duration = np.array(dc['routes'][pick]['legs'][0]['annotation']['duration'] + [0.0])
	print(np.sum(duration))
	
	df = pd.DataFrame({"locx":line[0],"locy":line[1],"dur":duration});
	df['dur'] = df['dur'].cumsum()/60.0

	df[['locx','locy']] = df[['locx','locy']].apply(reversetrans, axis=1);

	return df;






baseurl = "http://0.0.0.0:30606/route/v1/driving/"
tr = gettrip(c,baseurl,-111.850805,40.767031,-111.8808,40.777031);
print(tr);