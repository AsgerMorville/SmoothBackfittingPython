import numpy as np
from smooth_backfitting.plsbf import PLSBF, PLSBFLasso

"""Example usage of the partially linear smooth backfitting framework"""

np.random.seed(1)  # set seed for random generation

d = 5  # number of continuous covariates
m = 3  # dimension of linear covariate
n = 100  # number of observations

Z = np.random.uniform(size=(n,d))  # sampled continuous covariate points
W = np.random.uniform(size=(n,m))  # sampled linear covariate matrix
beta = np.random.uniform(size=(m,1))  # sampled coefficient vector


Y = np.cos(Z[:,[1]]) + W@beta + np.random.normal(size=n)

# We fit with the regular partially linear smooth backfitting approach
model = PLSBF()
model.fit(W=W, Z=Z, Y=Y)

# Find the fitted values
model.predict(W=W, Z=Z)

# Fit again but using the lasso-version.
model = PLSBFLasso(lmbda=0.25)
model.fit(W=W, Z=Z, Y=Y)

# Find the fitted values
model.predict(W=W, Z=Z)
