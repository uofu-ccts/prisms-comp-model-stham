#import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
import numpy as np;
from collections import Counter;
import datetime

def bzPatch(x1, y1, x2, y2, height):
	mid = (x2-x1)/2.0 + x1
	verts = [(x1,y1),(mid,y1),(mid,y2),(x2,y2),(x2,y2+height),(mid,y2+height),(mid,y1+height),(x1,y1+height),(x1,y1)]
	codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY,]
	return Path(verts,codes);

def backcheck(nodes, origins, dest):
# 	print(datetime.datetime.now().time())
	#sub = [b[0] for b in edges if b[1] == dest]
	if(dest in origins): #end of line
		return 0;
	out = 0;
	for i in nodes[dest][3]:
		if nodes[i][2] == 'a':
			out = max(out, backcheck(nodes,origins,i))
		else:
			out = max(out, nodes[i][2]);
	#print(dest,sub,out + 1);
	nodes[dest][2] = out + 1
	return nodes[dest][2];

def patchLayers(edges, width, height, gap):
	
	
	if len(edges) == 0:
		return [];
	
	annoyflag=False
	nodes = {}
	for i in edges:
		#weight update
		if(i[0] == i[1]):
			annoyflag = True;
			continue;
		if i[0] in nodes.keys():
			nodes[i[0]][0] += i[2];
		else:
			nodes[i[0]] = [i[2],0,0];
		if i[1] in nodes.keys():
			nodes[i[1]][1] += i[2];
			nodes[i[1]][2] += 1;
		else:
			nodes[i[1]] = [0,i[2],1];
			
	
	for i in nodes.keys():
		nodes[i] += [[b[0] for b in edges if b[1] == i]]
		nodes[i][2] = 'a'
		

	#stack index sorting
	nodecount = len(nodes);
	nodenames = list(nodes.keys());
	startnodes = [b[0] for b in nodes.items() if b[1][2] == 0]
	for i in startnodes:
		nodes[i][2] = 0;
	altnodes = [b[0] for b in nodes.items() if b[1][2] != 0]
	tcount = 0
	for i in altnodes:		
		nodes[i][2] = backcheck(nodes, startnodes, i);
		tcount += 1
	
	c = Counter([b[2] for b in nodes.values()]);
	maxpath = max(c.keys()) + 1
	maxstack = max(c.values())
	
	
# 	print(nodes)

	#stack height assign
	maxheight = 0.0
	for i in range(maxpath):
		s = [b[0] for b in nodes.items() if b[1][2] == i]
		sh = 0.0
		for ind,j in enumerate(s):
			temp1 = nodes[j][0]
			temp2 = nodes[j][1]
			nodes[j][0] = sh;
			nodes[j][1] = sh;
			sh += max(temp1,temp2) + gap
			
		maxheight = max(maxheight, sh);
	
# 	print(nodes)
	
# 	print(maxheight);
	heightconv = height / maxheight;
	widthconv = width / (maxpath)
	nodeadd = np.zeros((nodecount,2));
	
	mapedges = []
	for i in edges:
		if(i[2] > 0.0):
			x1 = nodes[i[0]][2] * widthconv;
			y1 = (nodes[i[0]][0] + nodeadd[nodenames.index(i[0])][0])*heightconv;
			x2 = nodes[i[1]][2] * widthconv
			y2 = (nodes[i[1]][0] + nodeadd[nodenames.index(i[1])][1])*heightconv;
			h = i[2]
			nodeadd[nodenames.index(i[0])][0] += h;
			nodeadd[nodenames.index(i[1])][1] += h;
	# 		print(x1,y1,x2,y2,h);
			bz = bzPatch(x1,y1,x2,y2,h*heightconv);
			mapedges += [PathPatch(bz, facecolor='grey', alpha=0.3, edgecolor='black')]
		
	nodeverts = []
	text = []
	for i in nodenames:
		w = (width / maxpath) * 0.1 
		x = nodes[i][2]*widthconv - (w/2)
		y = (nodes[i][0])*heightconv - (w/2)
		
		h = (max(nodeadd[nodenames.index(i)][0],nodeadd[nodenames.index(i)][1]))*heightconv + 0.1
		nodeverts += [Rectangle((x,y),w,h, facecolor='white')]
		text += [ ( x + w, y + (h/2), str(i) ) ] 
		
	if(annoyflag):
		print("Warning: one or more edges connect to the same node");

	#return PatchCollection(mapedges);
	
	return mapedges + nodeverts, text;


# # edges = [ ('a','c', 0.3), ('a','d',0.7), ('b','c',0.4), ('b','d',0.6), ('a','e',0.1), ('d','e',0.5) ];
# print("generating...")
# nodes = []
# for i in range(0,3*4,3):
# 	nodes += [ np.arange(0,3) + i ]
# 
# edges = []
# for i in range(len(nodes) - 1):
# 	for j in range(10):
# 		edges += [ (np.random.choice(nodes[i]),np.random.choice(nodes[i+1]),np.random.rand()) ]
# print(edges)
# print("processing...")
# 
# width = 10.0
# height = 4.0
# gap = 1.0
# pc, text = patchLayers(edges, width, height, gap);
# 
# print("plotting...")
# fig,ax = plt.subplots()
# ax.set_axis_off()
# ax.margins(0.01)
# patch = Rectangle((-1.0,-1.0),width+1.0, height+1.0,facecolor='white',edgecolor='none')
# ax.add_patch(patch)
# for i in pc:
# 	ax.add_patch(i);
# for i in text:
# 	plt.text(*i);
# 
# ax.set_xlim(-1.0, width + 1.0)
# ax.set_ylim(-1.0, height + 1.0)
# 
# plt.show()



	
	
	
		
	
	
	