import numpy as np
from smooth_backfitting.plsbf import PLSBF, PLSBFLasso
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True)

np.random.seed(1)

p = 5
n = 1000
Z = np.random.uniform(size=(n,p))-2
W = np.random.binomial(20, 0.4, (n,3))

beta1 = np.array([0, 0, 0])
beta2 = np.array([0, 0, -2.3])

Y = np.zeros((n,2))
Y[:,[0]] = np.expand_dims(np.cos(5*Z[:,0]) + np.random.normal(size=n,scale=0.15),1)
Y[:,[1]] = np.expand_dims(np.sin(5*Z[:,0]) + np.random.normal(size=n,scale=0.15),1)
Y[:, [0]] += np.expand_dims(W@beta1,1)
Y[:, [1]] += np.expand_dims(W@beta2,1)

#tester = PLSBF()
#tester.fit(W=W[:,[]], Z=Z, Y=Y)
#tester.fit(W=W, Z=Z[:,[]], Y=Y)
#preds = tester.predict(W=W, Z=Z[:,[]])
#plt.scatter(W[:,1],Y[:,0])
#plt.scatter(W[:,1], preds[:,0])
#plt.show()
#tester.fit(W=W[:, []], Z=Z[:,[]], Y=Y)
#tester.predict(W=W[:, []], Z=Z[:,[]])

tester2 = PLSBFLasso(lmbda=0.24)
#tester2.fit(W=W[:,[]], Z=Z, Y=Y)
#tester2.fit(W=W, Z=Z[:,[]], Y=Y)
#tester2.fit(W=W[:, []], Z=Z[:,[]], Y=Y)

W2 = np.random.binomial(20, 0.4, (n,6))
tester2.fit(W=W2, Z=Z, Y=Y, W_groups=[0, 0, 0, 1, 1, 1])
preds = tester2.predict(W=W2, Z=Z)
W2

tester = PLSBF()
import time
t0 = time.time()
for i in range(10):
    tester.fit(W=W, Z=Z, Y=Y)
#preds = tester.predict(W=W,Z=Z)
t1 = time.time()



#plt.scatter(x=Z[:,0], y=Y[:,0], color='blue')
#plt.scatter(x=Z[:,0], y=preds[:,0],  color='orange')
#plt.show()




#tester1 = PLSBFLasso(lmbda=0.4)
#tester1.fit(W=W, Z=Z, Y=Y)

#preds1 = tester1.predict(W=W,Z=Z)

#plt.scatter(x=Z[:,0], y=Y[:,0], color='blue')
#plt.scatter(x=Z[:,0], y=preds1[:,0],  color='orange')
#plt.show()

print(preds)
print("Time: ", (t1-t0)/10)

"""
tester2 = SmoothBackfittingLasso(lmbda=0.4)
tester2.fit(X=X, Y=Y)
preds2 = tester2.predict(X)

plt.scatter(x=X[:,0], y=Y[:,0], color='blue')
plt.scatter(x=X[:,0], y=preds2,  color='orange')
plt.show()"""