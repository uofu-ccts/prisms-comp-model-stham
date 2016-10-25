"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import sys

import mkl

mkl.set_num_threads(4)

def sigmoid(f): 
    return ( 1.0 / (1.0 + np.exp(-f)));

def revsigmoid(f):
	v = sigmoid(f);
	return (v*(1-v));

# data I/O
data = open(sys.argv[1], 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-0

# model parameters
Wxi = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whi = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Wci = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden

Wxf = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whf = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Wcf = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden

Wxo = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Who = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Wco = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden

Wxc = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whc = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden

Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output

bi = np.zeros((hidden_size, 1)) # hidden bias
bf = np.zeros((hidden_size, 1)) # hidden bias
bc = np.zeros((hidden_size, 1)) # hidden bias
bo = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev, cprev):
	"""
	inputs,targets are both list of integers.
	hprev is Hx1 array of initial hidden state
	returns the loss, gradients on model parameters, and last hidden state
	"""
	xs, hs, ys, ps = {}, {}, {}, {}
	iis, fs, cs, os = {}, {}, {}, {}
	cis = {}
	
	hs[-1] = np.copy(hprev)
	cs[-1] = np.copy(cprev)
	loss = 0
	# forward pass
	for t in range(len(inputs)):
		
		xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		
		#LSTM RNN
		iis[t] = sigmoid( np.dot( Wxi,xs[t] ) + np.dot( Whi,hs[t-1] ) + np.dot( Wci,cs[t-1] ) + bi )
		fs[t] = sigmoid( np.dot( Wxf,xs[t] ) + np.dot( Whf,hs[t-1] ) + np.dot( Wcf,cs[t-1] ) + bf )
		cis[t] = np.tanh( np.dot( Wxc,xs[t] ) + np.dot( Whc,hs[t-1] ) + bc )
		cs[t] = fs[t] * cs[t-1] + iis[t] * cis[t];
		os[t] = sigmoid( np.dot( Wxo,xs[t] ) + np.dot( Who,hs[t-1] ) + np.dot( Wco,cs[t] ) + bo )
		hs[t] = os[t] * np.tanh(cs[t]); 

		#standard RNN
		#hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
		loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
		
	#STUPID ANTIPATTERNS
	#backward pass: compute gradients going backwards
	dWxi, dWhi, dWci = np.zeros_like(Wxi), np.zeros_like(Whi), np.zeros_like(Wci)
	dWxf, dWhf, dWcf = np.zeros_like(Wxf), np.zeros_like(Whf), np.zeros_like(Wcf)
	dWxo, dWho, dWco = np.zeros_like(Wxo), np.zeros_like(Who), np.zeros_like(Wco)
	dWhy, dWxc, dWhc = np.zeros_like(Why), np.zeros_like(Wxc), np.zeros_like(Whc)
	
	dbi, dbf = np.zeros_like(bi), np.zeros_like(bf)
	dbc, dbo = np.zeros_like(bc), np.zeros_like(bo)
	dby = np.zeros_like(by);
	
	
	dhnext = np.zeros_like(hs[0]);
	dcnext = np.zeros_like(cs[0]);
	for t in reversed(range(len(inputs))):
		#yt
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		
		#ht
		dh = np.dot(Why.T, dy) + dhnext # backprop into h
		#dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
		do = np.tanh(cs[t]) * dh;
		dch = os[t] * dh;
		dchraw = (1 - cs[t] * cs[t]) * dch;
		
		#ot
		doraw = revsigmoid(do);
		#dxo = np.dot(Wxo.T, doraw); 
		dWxo += np.dot(doraw, xs[t].T);
		dohnext = np.dot(Who.T, doraw);
		dWho += np.dot(doraw, hs[t-1].T);
		dco = np.dot(Wco.T, doraw);
		dWco += np.dot(doraw, cs[t].T);
		dbo += doraw
		
		#ct
		dc = dchraw + dco + dcnext
		df = cs[t-1] * dc;
		dccnext = fs[t] * dc;
		di = cis[t] * dc;
		dcis = iis[t] * dc;
		dcisraw = (1 - cis[t] * cis[t]) * dcis;
		#dxc = np.dot(Wxc.T, dcisraw);
		dWxc += np.dot(dcisraw, xs[t].T);
		dchnext = np.dot(Whc.T, dcisraw);
		dWhc += np.dot(dcisraw, hs[t-1].T);
		dbc += dcisraw;		
		
		#ft
		dfraw = revsigmoid(df);
		#dxf = np.dot(Wxf.T, dfraw); 
		dWxf += np.dot(dfraw, xs[t].T);
		dfhnext = np.dot(Whf.T, dfraw);
		dWhf += np.dot(dfraw, hs[t-1].T);
		dfcnext = np.dot(Wcf.T, dfraw);
		dWcf += np.dot(dfraw, cs[t-1].T);
		dbf += dfraw
			
		#it  
		diraw = revsigmoid(di);
		#dxi = np.dot(Wxf.T, dfraw); 
		dWxi += np.dot(diraw, xs[t].T);
		dihnext = np.dot(Whi.T, diraw);
		dWhi += np.dot(diraw, hs[t-1].T);
		dicnext = np.dot(Wci.T, diraw);
		dWci += np.dot(diraw, cs[t-1].T);
		dbi += diraw
		
		#dbh += dhraw
		#dWxh += np.dot(dhraw, xs[t].T)
		#dWhh += np.dot(dhraw, hs[t-1].T)
		
		#finalize
		#dhnext = np.dot(Whh.T, dhraw)
		dhnext = dohnext + dchnext + dfhnext + dihnext 
		dcnext = dccnext + dfcnext + dicnext
		
	#for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
	for dparam in [dWxi, dWhi, dWci, dWxf, dWhf, dWcf, dWxo, dWho, dWco, dWhy, dWxc, dWhc, dbi, dbf, dbc, dbo, dby]:
		#np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		np.clip(dparam, -5, 5, out=dparam)
	#return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
	return loss, dWxi, dWhi, dWci, dWxf, dWhf, dWcf, dWxo, dWho, dWco, dWhy, dWxc, dWhc, dbi, dbf, dbc, dbo, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]

def sample(h, c, seed_ix, n):
	""" 
	sample a sequence of integers from the model 
	h is memory state, seed_ix is seed letter for first time step
	"""
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in range(n):
	#     h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
	#     y = np.dot(Why, h) + by
	#     p = np.exp(y) / np.sum(np.exp(y))
		#LSTM RNN
		ii = sigmoid( np.dot( Wxi,x ) + np.dot( Whi,h ) + np.dot( Wci,c ) + bi )
		f = sigmoid( np.dot( Wxf,x ) + np.dot( Whf,h ) + np.dot( Wcf,c ) + bf )
		ci = np.tanh( np.dot( Wxc,x ) + np.dot( Whc,h ) + bc )
		c = f * c + ii * ci;
		o = sigmoid( np.dot( Wxo,x ) + np.dot( Who,h ) + np.dot( Wco,c ) + bo )
		h = o * np.tanh(c); 
		
		#standard RNN
		#hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		
		y = np.dot(Why, h) + by # unnormalized log probabilities for next chars
		p = np.exp(y) / np.sum(np.exp(y)) # probabilities for next chars
		
		
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		#ix = np.argmax(p);
		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)
    
	#print(np.reshape(p,(vocab_size)));
	return ixes

n, p = 0, 0
mWxi, mWhi, mWci = np.zeros_like(Wxi), np.zeros_like(Whi), np.zeros_like(Wci)
mWxf, mWhf, mWcf = np.zeros_like(Wxf), np.zeros_like(Whf), np.zeros_like(Wcf)
mWxo, mWho, mWco = np.zeros_like(Wxo), np.zeros_like(Who), np.zeros_like(Wco)
mWhy, mWxc, mWhc = np.zeros_like(Why), np.zeros_like(Wxc), np.zeros_like(Whc)
#mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbi, mbf, mbc, mbo, mby = np.zeros_like(bi),np.zeros_like(bf),np.zeros_like(bc),np.zeros_like(bo),np.zeros_like(by)
#mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    cprev = np.zeros((hidden_size,1))
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, cprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  #loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  loss, dWxi, dWhi, dWci, dWxf, dWhf, dWcf, dWxo, dWho, dWco, dWhy, dWxc, dWhc, dbi, dbf, dbc, dbo, dby, hprev, cprev = lossFun(inputs,targets,hprev,cprev);
  
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: 
  	print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  	sys.stdout.flush()
  
  # perform parameter update with Adagrad
#   for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
#                                 [dWxh, dWhh, dWhy, dbh, dby], 
#                                 [mWxh, mWhh, mWhy, mbh, mby]):
  	
  for param, dparam, mem in zip([Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxo, Who, Wco, Why, Wxc, Whc, bi, bf, bc, bo, by], 
                                [dWxi, dWhi, dWci, dWxf, dWhf, dWcf, dWxo, dWho, dWco, dWhy, dWxc, dWhc, dbi, dbf, dbc, dbo, dby], 
                                [mWxi, mWhi, mWci, mWxf, mWhf, mWcf, mWxo, mWho, mWco, mWhy, mWxc, mWhc, mbi, mbf, mbc, mbo, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  
