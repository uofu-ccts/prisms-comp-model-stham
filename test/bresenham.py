import numpy as np;
import matplotlib.pyplot as plt;





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
	return out,(x1,y1);



# print(line)


# line = [(0,0),(1,1),(2,2),(3,3)]
for k in range(0,30):
	outarr = np.zeros((100,100))
	for j in range(0,10):
		t = np.random.randint(100,size=4)
		line,s = bresenham(*t);
		print(t, line,s)
		for i in line:
			outarr[i[0],i[1]] = 1
			outarr[s[0],s[1]] = -1

	plt.matshow(outarr)
	plt.show();
