import numpy as np;
import matplotlib.pyplot as plt;

a = np.random.randn(100,100);

b = np.triu(a);



c = b + b.T - np.diag(b.diagonal());

plt.imshow(c,cmap="viridis");
plt.show();