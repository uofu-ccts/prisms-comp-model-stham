#import pandas as pd;
#import time;
#import matplotlib.pyplot as plt;
#import matplotlib.cm as cm
#import matplotlib.patches as mpatch;
import numpy as np;
import collections

def Tree():
	return collections.defaultdict(Tree);

#adds sequences to the list

def TreeAdd(tree, sequence):
	
	prevnode = tree
	if(None in prevnode): prevnode[None] += 1;
	else: prevnode[None] = 1;
	
	for i in sequence:
		node = prevnode[i];
		if(None in node): node[None] += 1;
		else: node[None] = 1;
		prevnode = node;
		
#gets all possible sequences, including short sequences

def TreeTraverse(tree):
	#print(tree.keys())
	x = []
	for i in tree.keys():
		if(i != None):
			#y is a list -or- a list of lists
			y = TreeTraverse(tree[i]);
			if(len(y) == 0):
				y = [i];
				x.append(y);
			else:	
				for j in y:
					j.append(i);
					x.append(j);
	return x;


def TreePick(tree):
	x = []
	k = list(tree.keys())
	k.remove(None)
	if(len(k) > 0):
		p = np.array([tree[b][None] for b in k])
		k.append(None)
		p = np.append(p,tree[None] - np.sum(p))
		p = p / np.sum(tree[None])
		choice = np.random.choice(k,p=p);
		if(choice == None):
			return x;
		x += TreePick(tree[choice]);
		x += [choice]
	return x;
	
	
def ReverseAll(l):
	for i in l:
		i.reverse();
	
def Seqcompare(a,b):
	#diff = np.subtract.outer(a,b);
	#diffmask = 1 - np.clip(np.abs(diff),0,1)
	diffmask = np.equal.outer(a,b)
	map = diffmask * np.multiply.outer(range(len(a)),range(len(b)))
	mask = [];
	#reduce array
	h = map;
	while(len(h) > 0):
		ind = np.unravel_index(np.argmax(h),h.shape)
		if(all(ind)): 
			mask += [ind]
		h = map[:ind[0],:ind[1]]
	
	mask.reverse();
	com1, com2 = [], [];
	for i in range(len(mask) - 1):
		com1 += [a[mask[i][0]]]
		com2 += [b[mask[i][1]]]
		if( (mask[i+1][0] - mask[i][0]) > 1 ):
			com1 += [True];
		if( (mask[i+1][1] - mask[i][1]) > 1 ):
			com2 += [True];
	if(len(mask) >= 1):
		com1 += [a[mask[len(mask)-1][0]]]
		com2 += [b[mask[len(mask)-1][1]]]	
# 	com1 = [a[i[0]] for i in mask]
# 	com2 = [b[i[1]] for i in mask]
	
	return com1, com2;

def MergeTrees(a,b):
	out = Tree();
	out[None] = a[None] + b[None];
	allkeys = list(set(list(a.keys()) + list(b.keys())))
	for i in allkeys:
		if i == None: continue;
		if (i in a) and (i in b):
			out[i] = MergeTrees(a[i],b[i]);
		elif i in a:
			out[i] = a[i];
		elif i in b:
			out[i] = b[i];
	
	
	return out;

def CollapseTrees(trees):
	newtrees = []
	if(len(trees) == 1):
		return trees[0];
	for i in range(len(trees)):
		if i % 2 == 0:
			a, b = i, i+1
			if(b < len(trees)):
				newtrees += [MergeTrees(trees[a],trees[b])]
			else:
				newtrees += [trees[a]]
	return CollapseTrees(newtrees);

def TreeCollapseTest():	
	trees = []
	for i in range(11):
		tr = Tree()
		a = np.random.randint(0,10,(4,4))
		for j in a:
			TreeAdd(tr,j);
		trees += [tr];
		
	s = CollapseTrees(trees);
	print(s[None])
	print(sum([b[None] for b in trees]))
	print(s)
	return s;

	
# tr = Tree();
# 
# for i in range(100):
# 	a = np.random.randint(0,400,(10));
# 	b = np.random.randint(0,400,(10));
# 	s1, s2 = Seqcompare(a,b);
# 	TreeAdd(tr, s1);
# 	TreeAdd(tr, s2);
# 
# for i in range(10):
# 	x = TreePick(tr);
# 	x.reverse()
# 	print(x);
# 
# out = TreeTraverse(tr);
# ReverseAll(out);
# print(len(out));
# print(len(tr));


	
	

