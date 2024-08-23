import numpy as np
from smooth_backfitting.grid_data import nadaraya_watson_estimator
from scipy.optimize import minimize
from smooth_backfitting.additive_function import AdditiveFunction


# Numba functions
import numba as nb

@nb.jit('(float64[:,], float64)', fastmath=True)
def integrate_1d_fast(vector, dz):
    return (2 * vector.sum() - vector[0] - vector[-1]) * dz / 2

@nb.jit('(float64[:,:],float64[:,:])', fastmath=True)
def check_sbf_conv_fast(m_old, m_new):
    eps = 0.00001
    for j in range(m_old.shape[1]):
        if np.sum(np.power(m_old[:, j] - m_new[:, j], 2)) / (np.sum(np.power(m_old[:, j], 2)) + eps) > eps:
            return False
    return True

@nb.jit('(float64[:,:,:],float64[:,:], float64[:,:,:,:], float64, float64)', fastmath=True)
def smooth_backfitting_loop_fast(nw_est, p_hat_1d, p_hat_2d, dz, maxIter):
    M, d, y_dim = nw_est.shape
    m_hat = np.zeros((M, d, y_dim))
    for b in range(y_dim):
        for t in range(maxIter):
            m_hat_j_old = np.copy(m_hat[:, :, b])
            for j in range(d):
                for l in range(M):
                    integral_sum = 0
                    for k in range(d):
                        if k != j:
                            integrand = m_hat[:, k, b] * p_hat_2d[l, :, j, k]
                            integral_sum += integrate_1d_fast(integrand, dz)
                    m_hat[l, j, b] = nw_est[l, j, b] - integral_sum / p_hat_1d[l, j]
            if check_sbf_conv_fast(m_hat_j_old, m_hat[:, :, b]):
                break
    return m_hat


def smooth_backfitting_loop_numba(y_center, grid):
    nw_est = nadaraya_watson_estimator(y=y_center, p_hat=grid.p_hat_1d, kernel_evals=grid.kernel_evals)
    return smooth_backfitting_loop_fast(nw_est=nw_est,p_hat_1d=grid.p_hat_1d, p_hat_2d=grid.p_hat_2d,
                                        dz=grid.dz, maxIter=grid.maxIter)


def integrate_1d(vector, dz):
    return (2 * vector.sum() - vector[0] - vector[-1]) * dz / 2


def m_hat_update(grid, nw_est, m_hat_old):
    M, d = m_hat_old.shape
    m_hat = np.copy(m_hat_old)
    for j in range(d):
        for l in range(M):
            integral_sum = 0
            for k in range(d):
                if k != j:
                    integrand = m_hat[:, k] * grid.p_hat_2d[l, :, j, k]
                    integral_sum += integrate_1d(integrand, grid.dz)
            m_hat[l, j] = nw_est[l, j] - integral_sum / grid.p_hat_1d[l, j]
    return m_hat


def m_hat_update_lasso(grid, nw_est, m_hat_old, lmbda):
    M, d = m_hat_old.shape
    m_hat = np.copy(m_hat_old)
    normalizing_const = np.ones(d)
    for j in range(d):
        Pi_minus_j = np.zeros(M)
        for l in range(M):
            integral_sum = 0
            for k in range(d):
                if k != j:
                    integrand = m_hat[:, k] * grid.p_hat_2d[l, :, j, k]
                    integral_sum += integrate_1d(integrand, grid.dz)
            Pi_minus_j[l] = nw_est[l, j] - integral_sum / (grid.p_hat_1d[l, j])
        # Calculate norm of Pi_minus_j:
        integrand = np.power(Pi_minus_j, 2) * grid.p_hat_1d[:, j]
        Pi_minus_j_norm = np.sqrt(integrate_1d(integrand, grid.dz))
        normalizing_const[j] = np.maximum(0, 1 - lmbda / Pi_minus_j_norm)
        m_hat[:, j] = normalizing_const[j] * Pi_minus_j
    return m_hat


def smooth_backfitting_lasso_loop(y_center, grid, lmbda):
    y_dim = y_center.shape[1]
    m_hat = np.zeros((grid.M, grid.d, y_dim))
    nw_est = nadaraya_watson_estimator(y=y_center, p_hat=grid.p_hat_1d, kernel_evals=grid.kernel_evals)
    for j in range(y_dim):
        for l in range(grid.maxIter):
            m_hat_j_old = np.copy(m_hat[:, :, j])
            m_hat[:, :, j] = m_hat_update_lasso(grid=grid, nw_est=nw_est[:, :, j],
                                                m_hat_old=m_hat[:, :, j], lmbda=lmbda)
            if check_sbf_conv(m_hat_j_old, m_hat[:, :, j]):
                break
    return m_hat



def smooth_backfitting_loop(y_center, grid):
    y_dim = y_center.shape[1]
    m_hat = np.zeros((grid.M, grid.d, y_dim))
    nw_est = nadaraya_watson_estimator(y=y_center, p_hat=grid.p_hat_1d, kernel_evals=grid.kernel_evals)
    for j in range(y_dim):
        for l in range(grid.maxIter):
            m_hat_j_old = np.copy(m_hat[:, :, j])
            m_hat[:, :, j] = m_hat_update(grid=grid, nw_est=nw_est[:, :, j],
                                          m_hat_old=m_hat[:, :, j])
            if check_sbf_conv(m_hat_j_old, m_hat[:, :, j]):
                break
    return m_hat


def plsbf_lasso_loop(y_center, grid, lmbda):
    y_dim = y_center.shape[1]
    m_hat = np.zeros((grid.M, grid.d, y_dim))
    beta_hat = np.zeros((grid.m, y_dim))
    for j in range(y_dim):
        for l in range(grid.maxIter):
            # We update self.m_hat and self.beta with a coordinate-descent-type of optimization
            # Save the current m_hat estimate
            m_hat_j_old = np.copy(m_hat[:, :, j])
            beta_hat_old = np.copy(beta_hat[:, j])

            # For a fixed beta, update the m_hat functions
            new_y = y_center[:, [j]] - grid.w_standardized @ beta_hat[:, [j]]
            nw_est = np.squeeze(nadaraya_watson_estimator(y=new_y, p_hat=grid.p_hat_1d,
                                                          kernel_evals=grid.kernel_evals))
            m_hat[:, :, j] = m_hat_update_lasso(grid=grid, nw_est=nw_est, m_hat_old=m_hat[:, :, j], lmbda=lmbda)

            # For a fixed m_hat, update the beta-parameter
            y_tilde = y_tilde_calculate(y_center=y_center[:, j], grid=grid, m_hat=m_hat[:, :, j])
            beta_hat[:, j] = beta_group_update(grid=grid, beta_old=beta_hat[:, j], y=y_tilde, lmbda=lmbda)

            # Check for convergence
            if check_sbf_conv(m_hat_j_old, m_hat[:, :, j]) and check_beta_conv(beta_hat_old, beta_hat[:, j]):
                break
    return m_hat, beta_hat

def plsbf_lasso_loop_sep(y_center, grid, lmbdaW, lmbdaZ):
    y_dim = y_center.shape[1]
    m_hat = np.zeros((grid.M, grid.d, y_dim))
    beta_hat = np.zeros((grid.m, y_dim))
    for j in range(y_dim):
        for l in range(grid.maxIter):
            # We update self.m_hat and self.beta with a coordinate-descent-type of optimization
            # Save the current m_hat estimate
            m_hat_j_old = np.copy(m_hat[:, :, j])
            beta_hat_old = np.copy(beta_hat[:, j])

            # For a fixed beta, update the m_hat functions
            new_y = y_center[:, [j]] - grid.w_standardized @ beta_hat[:, [j]]
            nw_est = np.squeeze(nadaraya_watson_estimator(y=new_y, p_hat=grid.p_hat_1d,
                                                          kernel_evals=grid.kernel_evals))
            m_hat[:, :, j] = m_hat_update_lasso(grid=grid, nw_est=nw_est, m_hat_old=m_hat[:, :, j], lmbda=lmbdaZ)

            # For a fixed m_hat, update the beta-parameter
            y_tilde = y_tilde_calculate(y_center=y_center[:, j], grid=grid, m_hat=m_hat[:, :, j])
            beta_hat[:, j] = beta_group_update(grid=grid, beta_old=beta_hat[:, j], y=y_tilde, lmbda=lmbdaW)

            # Check for convergence
            if check_sbf_conv(m_hat_j_old, m_hat[:, :, j]) and check_beta_conv(beta_hat_old, beta_hat[:, j]):
                break
    return m_hat, beta_hat



def linear_regression_group_lasso(grid, Y, lmbda):
    y_dim = Y.shape[1]
    y_center = Y - Y.mean(axis=0)
    beta_hat = np.zeros((grid.m, y_dim))
    for j in range(y_dim):
        for l in range(grid.maxIter):
            beta_old = np.copy(beta_hat[:, j])
            beta_hat[:, j] = beta_group_update(grid=grid, beta_old=beta_hat[:, j], y=y_center[:, j], lmbda=lmbda)
            if check_beta_conv(beta_hat[:, j], beta_old):
                break
    return beta_hat


def y_tilde_calculate(y_center, grid, m_hat):
    y_tilde = np.copy(y_center)
    for i in range(grid.n):
        for j in range(grid.d):
            y_tilde[i] -= integrate_1d(m_hat[:, j] * grid.kernel_evals[:, i, j], grid.dz)
    return y_tilde


def beta_group_update(grid, beta_old, y, lmbda):
    # This function updates beta in a group-wise fashion assuming the covariates are grid.w_center
    # The number of groups is given by the number grid.L
    beta = np.copy(beta_old)
    for l in range(grid.L):
        l_indices = grid.index_map[l]
        beta[l_indices] = beta_l_update(l_index=l, beta=beta, y=y, grid=grid, lmbda=lmbda)
    return beta


def beta_l_update(l_index, beta, y, grid, lmbda):
    # We check first if beta_l is zero.
    l_indices = grid.index_map[l_index]
    indices_minus_l = [i for i in range(grid.m) if i not in l_indices]
    residual = np.copy(y)
    residual -= grid.w_standardized[:, indices_minus_l] @ beta[indices_minus_l]
    if np.sqrt((1 / grid.n) * np.sum(np.power(grid.w_standardized[:, l_indices].T @ residual, 2))) < lmbda:
        return np.zeros(len(l_indices))
    else:
        # In this case, the solution of beta_l is not zero.
        return NewtonSolver(Y=y, Z=grid.w_standardized[:, l_indices],
                            theta=beta[l_indices], lmbda=lmbda)


def NewtonSolverOld(Y, Z, theta, lmbda):
    m = Z.shape[1]
    n = Y.shape[0]
    for l in range(m):
        def f(theta_l):
            return 0.5 * (1 / n) * np.sum((Y - Z @ theta + Z[:, l] * theta[l] - Z[:, l] * theta_l) ** 2) \
                + lmbda * np.sqrt(theta.shape[0]) * np.power(
                    np.sum(np.power(theta, 2)) - theta[l] ** 2 + theta_l ** 2, 0.5)

        x0 = np.array([0])
        theta_l = minimize(f, x0, method='BFGS')
        theta[l] = theta_l.x[0]
    return theta

def NewtonSolver(Y, Z, theta, lmbda):
    m = Z.shape[1]
    n = Y.shape[0]
    norm_factor = 1 / n
    sqrt_lambda = lmbda * np.sqrt(theta.shape[0])

    # Precompute Z @ theta and reuse
    Ztheta = Z @ theta
    Z2_sum = np.sum(Z ** 2, axis=0)  # Precompute the sum of squares of Z's columns

    for l in range(m):
        Z_l = Z[:, l]
        Ztheta_minus_Zl_thetal = Ztheta - Z_l * theta[l]

        def f(theta_l):
            residual = Y - Ztheta_minus_Zl_thetal + Z_l * theta_l
            penalty = np.sqrt(np.sum(theta ** 2) - theta[l]**2 + theta_l**2)
            return 0.5 * norm_factor * np.sum(residual ** 2) + sqrt_lambda * penalty

        # Initial guess could be the previous value of theta[l]
        theta_l = minimize(f, np.array([theta[l]]), method='BFGS').x[0]
        # Update theta and Ztheta accordingly
        Ztheta += Z_l * (theta_l - theta[l])
        theta[l] = theta_l

    return theta


def plsbf_loop(y_center, grid):
    y_dim = y_center.shape[1]
    beta = np.zeros((grid.m, y_dim))
    m_hat = np.zeros((grid.M, grid.d, y_dim))
    z_grid = grid.z_grid  # This is the original grid of points (not the one on the unit hyperplane)

    m_hat_w = smooth_backfitting_loop_numba(y_center=grid.w_center, grid=grid)
    m_hat_y = smooth_backfitting_loop_numba(y_center=y_center, grid=grid)

    additive_w = AdditiveFunction(z_grid_points=z_grid, m_points=m_hat_w, y_mean=0)
    additive_y = AdditiveFunction(z_grid_points=z_grid, m_points=m_hat_y, y_mean=0)

    Y_tilde = y_center - additive_y.predict(grid.z_observations)
    W_tilde = grid.w_center - additive_w.predict(grid.z_observations)
    for j in range(y_dim):
        beta[:, j] = np.squeeze(
            np.linalg.solve(W_tilde.T @ W_tilde, W_tilde.T @ np.expand_dims(Y_tilde[:, j], 1)))
        for l in range(grid.M):
            for k in range(grid.d):
                m_hat[l, k, j] = m_hat_y[l, k, j] - np.dot((m_hat_w[l, k, :]), beta[:, j])
    return m_hat, beta


def linear_regression(W, Y):
    return np.linalg.solve(W.T @ W, W.T @ Y)


def check_sbf_conv(m_old, m_new):
    eps = 0.00001
    for j in range(m_old.shape[1]):
        if np.sum(np.power(m_old[:, j] - m_new[:, j], 2)) / (np.sum(np.power(m_old[:, j], 2)) + eps) > eps:
            return False
    return True

def check_beta_conv(beta_old, beta_new):
    eps = 0.00001
    if np.linalg.norm(beta_old - beta_new, 2) > eps:
        return False
    else:
        return True

#############################
#############################
#############################
#############################





"""#@nb.jit('(float64[:,:],float64[:,:,:,:], float64[:,:], float64, float64, float64)', fastmath=True)
def full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par, maxIter):
    M, d = p_hat_tab.shape
    m_hat = np.zeros((M, d))
    for r in range(maxIter):
        m_old = np.copy(m_hat)
        m_hat = opt_loop(m_hat, p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par)
        if checkconv(m_old, m_hat, d):
            # print("convergence at: iteration=", r)
            break
    return m_hat


def opt_loop(m_hat, p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par):
    M, d = m_hat.shape
    normalizing_const = np.ones(d)
    for j in range(d):
        Pi_minus_j = np.zeros(M)
        for l in range(M):
            integral_sum = 0
            for k in range(d):
                if k != j:
                    integrand = m_hat[:, k] * p_hat_tab2[l, :, j, k]
                    integral_sum += integrate_0_1(integrand, dx)
            Pi_minus_j[l] = f_hat_tab[l, j] - integral_sum / (p_hat_tab[l, j])
        # Calculate norm of Pi_minus_j:
        integrand = np.power(Pi_minus_j, 2) * p_hat_tab[:, j]
        Pi_minus_j_norm = np.sqrt(integrate_0_1(integrand, dx))
        normalizing_const[j] = np.maximum(0, 1 - lambda_par / Pi_minus_j_norm)
        m_hat[:, j] = normalizing_const[j] * Pi_minus_j

    return m_hat"""
