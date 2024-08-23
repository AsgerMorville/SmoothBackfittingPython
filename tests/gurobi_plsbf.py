import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from smooth_backfitting.grid_data import GridDataPLSBF,nadaraya_watson_estimator, standardize
from smooth_backfitting.kernel import EpanKernel
from smooth_backfitting.plsbf import PLSBF, PLSBFLasso
np.set_printoptions(precision = 4, suppress = True)

def integrate(vector, dx):
    vecSum = gp.quicksum(vector)
    return (2 * vecSum - vector[0] - vector[-1]) * dx / 2

def integrate2d(array, arrLength, dx):
    arrSum = 4 * gp.quicksum(array)
    for l in range(arrLength):
        arrSum -= 2*array[0,l] + 2*array[-1,l] + 2*array[l,0] + 2*array[l,-1]
    return arrSum * dx**2 / 4

def GurobiOptLoop(W, Z, Y, lmbda):
    """
    This functions finds mHat, betaHat
    :param gridData:
    :return:
    """
    grid_obj = GridDataPLSBF(W=W, Z=Z, baseline_kernel=EpanKernel(), W_groups=[i for i in range(W.shape[1])])

    y_center = Y-Y.mean(axis=0)
    M = grid_obj.M
    d_c = grid_obj.d
    d_d = grid_obj.m
    Xd = grid_obj.w_standardized
    n = grid_obj.n

    f_hat = np.squeeze(nadaraya_watson_estimator(y=y_center, p_hat=grid_obj.p_hat_1d, kernel_evals=grid_obj.kernel_evals))
    NW_W_obj = nadaraya_watson_estimator(y=Xd, p_hat=grid_obj.p_hat_1d, kernel_evals=grid_obj.kernel_evals)

    model = gp.Model("PLSBF Lasso")

    g = model.addMVar((M, d_c), lb=-1000, ub=1000, name="g")
    beta = model.addMVar(d_d, lb=-1000, ub=1000, vtype=gp.GRB.CONTINUOUS, name="beta")
    z = model.addMVar((M - 1, d_c), lb=-1000, ub=1000, name="z")
    w = model.addMVar(d_c, lb=-1000, ub=1000, name="w")
    one_norm_beta = model.addVar(lb=-1000, ub=1000, name="w")
    sqrtdx = np.sqrt(grid_obj.dx)
    for j in range(d_c):
        for l in range(M - 1):
            model.addConstr(z[l, j] == sqrtdx * (g[l, j] + g[l + 1, j]) / 2, f"normconstr")
        model.addConstr(w[j] == gp.norm(z[:, j], 2))
    model.addConstr(one_norm_beta == gp.norm(beta, 1))
    terms = []

    # T1 is the term involving the sum of squares
    terms.append(0.5 * gp.quicksum((y_center[i] - gp.quicksum(Xd[i,l] * beta[l] for l in range(d_d))) * (y_center[i] - gp.quicksum(Xd[i,l] * beta[l] for l in range(d_d))) / n for i in range(n)))

    # Now we add the T2 terms
    # Diagonal terms of 2nd term
    for j in range(d_c):
        integral = integrate(g[:, j] * g[:, j] * grid_obj.p_hat_1d[:, j], grid_obj.dx)
        terms.append(0.5 * integral)

    # Cross terms of 2nd term
    for j in range(d_c):
        for k in range(j + 1, d_c):
            integral = gp.quicksum(
                4 * g[l, j] * g[u, k] * grid_obj.p_hat_2d[l, u, j, k] for l in range(M) for u in range(M))
            edges = gp.quicksum(2 * g[l, j] * g[0, k] * grid_obj.p_hat_2d[l, 0, j, k]
                                + 2 * g[l, j] * g[-1, k] * grid_obj.p_hat_2d[l, -1, j, k]
                                + 2 * g[0, j] * g[l, k] * grid_obj.p_hat_2d[0, l, j, k]
                                + 2 * g[-1, j] * g[l, k] * grid_obj.p_hat_2d[-1, l, j, k] for l in range(M))

            terms.append(0.5 * 2 * (integral - edges) * grid_obj.dx ** 2 / 4)

    # Finally we add the T3 term
    for j in range(d_c):
        # fHatBeta = gp.quicksum(grid_obj.fHat[l] - grid_obj.X_NW_table[l,:]@beta for l in range(grid_obj.M))
        #integral = integrate(vector=g[:, j] * (grid_obj.fHat[:, j] - grid_obj.X_NW_table[:, j, 0] * beta[0] - grid_obj.X_NW_table[:, j, 1] * beta[1]) * grid_obj.pHat[:, j], dx=grid_obj.dx)
        integral = integrate(vector=g[:, j] * (
                f_hat[:, j] - gp.quicksum(
            NW_W_obj[:, j, l] * beta[l] for l in range(d_d))) * grid_obj.p_hat_1d[:, j], dx=grid_obj.dx)
        terms.append(- integral)

    # Penalizing terms
    lmbdaPar = lmbda
    terms.append(lmbdaPar * one_norm_beta)
    for j in range(d_c):
        terms.append(lmbdaPar * w[j])

    for j in range(d_c):
        integral = integrate(vector=g[:, j] * grid_obj.p_hat_1d[:, j], dx=grid_obj.dx)
        model.addConstr(integral == 0)

    model.setObjective(gp.quicksum(terms), gp.GRB.MINIMIZE)

    model.optimize()
    mHat = g.X
    betaHat = beta.X
    return mHat, betaHat

np.random.seed(2)
# grid_obj, init_obj, f_hat, lmbda
n = 200
d_d = 2
d_c = 3

Z = np.random.uniform(low=np.zeros(d_c),high=np.ones(d_c),size=(n,d_c))
W = np.random.binomial(n=1, p=0.5, size=(n, d_d))
W = standardize(W)

beta0 = np.zeros(d_d)
beta0[1] = -2

Y = np.expand_dims(np.cos(4 * Z[:,0]) + np.sin(4 * Z[:,1]) + beta0[1]*W[:,1] + np.random.normal(scale=0.1, size=n),1)

lmbda = 0.14

m_hat, beta = GurobiOptLoop(W=W, Z=Z, Y=Y, lmbda=lmbda)

model1 = PLSBF()
model1.fit(W=W, Z=Z, Y=Y)
m_hat_est = model1.additive_function.m_points
beta_hat_est = model1.additive_function.beta

model2 = PLSBFLasso(lmbda=lmbda)
model2.fit(W=W, Z=Z, Y=Y)
m_hat_est2 = model2.additive_function.m_points
beta_hat_est2 = model2.additive_function.beta

print("TEST DIFF PLSBF, beta: ", np.mean(np.abs(np.squeeze(beta_hat_est)-beta)))
print("TEST DIFF PLSBFLasso, beta: ", np.mean(np.abs(np.squeeze(beta_hat_est2)-beta)))

print("TEST DIFF PLSBF, mhat: ", np.mean(np.abs(np.squeeze(m_hat_est)-m_hat)))
print("TEST DIFF PLSBFLasso, mhat: ", np.mean(np.abs(np.squeeze(m_hat_est2)-m_hat)))




index = 0
plt.scatter(range(m_hat.shape[0]),m_hat[:,index], color="blue")
plt.scatter(range(m_hat.shape[0]),m_hat_est[:,index,0], color="pink")
plt.scatter(range(m_hat.shape[0]),m_hat_est2[:,index,0], color = "orange")
plt.show()

