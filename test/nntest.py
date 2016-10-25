import numpy as np;
import matplotlib.pyplot as plt;
import mkl;

mkl.set_num_threads(4)

N = 100;
D = 2;
K = 3;

X = np.zeros((N*K,D));
Y = np.zeros(N*K, dtype='uint8');
for j in range(K):
	ix = list(range(N*j,N*(j+1)));
	r = np.linspace(0.0,1,N);
	t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
	X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	Y[ix] = j
	
# plt.scatter(X[:,0],X[:,1],c=Y,s=40, cmap=plt.cm.Spectral)
# plt.show()
# 
# exit()

#softmax implementation

W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(200):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[list(range(num_examples)),Y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[list(range(num_examples)),Y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db


scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == Y)))



#neural net implementation
print("\n\nNEURAL NET PARTY\n")

h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
vW = np.zeros_like(W)
mW = np.zeros_like(W)
b = np.zeros((1,h))
vb = np.zeros_like(b)
mb = np.zeros_like(b)
W2 = 0.01 * np.random.randn(h,K)
vW2 = np.zeros_like(W2)
mW2 = np.zeros_like(W2)
b2 = np.zeros((1,K))
vb2 = np.zeros_like(b2)
mb2 = np.zeros_like(b2)

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

nr,nc=W.shape;
extent = [-0.5, nc-0.5, nr-0.5, -0.5]

Wcollect = []; 
mu = 0.;
# vW = 0.0; vW2 = 0.0; vb = 0.0; vb2 = 0.0;
# mW = 0.0; mW2 = 0.0; mb = 0.0; mb2 = 0.0;
beta1 = 0.9; beta2 = 0.999; eps = 1e-8;

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),Y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 100 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),Y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W
  
  #perform a parameter update
#   for p, dp, m, v in zip([W,b,W2,b2],[dW,db,dW2,db2],[mW,mb,mW2,mb2],[vW,vb,vW2,vb2]):
#     print(p);
#     m = beta1*m + (1-beta1)*dp;
#     v = beta2*v + (1-beta2)*(dp**2)
#     p += -step_size * m / (np.sqrt(v) + eps)
#     print(p)
  vW = mu * vW - step_size * dW
  W += vW
  vb = mu * vb - step_size * db
  b += vb
  vW2 = mu * vW2 - step_size * dW2
  W2 += vW2
  vb2 = mu * vb2 - step_size * db2
  b2 += vb2

  if i% 100 == 0:
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == Y)))
    
  if i%500 == 0:
  	Wcollect += [np.copy(W)];
  	#ax = plt.subplot(21,1,int(i/500))
  	#plt.imshow(W,cmap=plt.cm.gray,extent=extent,interpolation='nearest');



hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == Y)))


for ind,i in enumerate(Wcollect):
	ax = plt.subplot(21,1,ind+1);
	ax.imshow(i,cmap=plt.cm.Spectral,extent=extent,interpolation='nearest');
	
plt.subplot(21,1,21)
ax.imshow(W,cmap=plt.cm.Spectral,extent=extent,interpolation='nearest');
plt.show();
exit()

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# plot the resulting classifier



plt.show()
