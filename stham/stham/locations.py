import pyproj

#does this global stuff work?
latlongproj = pyproj.Proj(init='epsg:4326');
xyproj = pyproj.Proj(init='epsg:26912');

def latlongtoxy(long,lat):
	locx,locy=pyproj.transform(latlongproj,xyproj,long,lat);
	return locx,locy

def xytolatlong(locx,locy):
	long,lat=pyproj.transform(xyproj,latlongproj,locx,locy);
	return long,lat

def setproj(latlong,xy):
	global latlongproj,xyproj
	latlongproj = pyproj.Proj(init=latlong);
	xyproj = pyproj.Proj(init=xy);


def runit():
	x,y = latlongtoxy(129.7,32.58)
	print(x,y)
	setproj('epsg:4326','epsg:3968')
	x,y = latlongtoxy(129.7,32.58)
	print(x,y)
	long,lat = xytolatlong(x,y)
	print(long,lat)

if __name__ == '__main__':
	runit();