import numpy as np
from smooth_backfitting.kernel import kernel_h

class GridDataSBF:
    def __init__(self, Z, baseline_kernel, bandwidth=None):
        self.M = 100
        self.d = Z.shape[1]
        self.n = Z.shape[0]
        self.dz = 1 / (self.M - 1)

        Z_min = np.min(Z, axis=0)
        Z_max = np.max(Z, axis=0)
        self.z_unit_grid = np.linspace(np.zeros(self.d), np.ones(self.d), self.M)
        self.z_grid = np.linspace(Z_min, Z_max, self.M)
        self.z_obs_unitized = (Z - Z_min) / (Z_max - Z_min)
        self.z_observations = Z
        if bandwidth is not None:
            self.bandwidth = np.ones(self.d) * bandwidth
        else:
            self.bandwidth = initialize_h(Z=self.z_obs_unitized)
        # Precompute tables
        self.kernel_evals = Kh_table_generator(z=self.z_obs_unitized, z_grid=self.z_unit_grid, kernel=baseline_kernel,
                                               h=self.bandwidth)
        self.p_hat_1d = phat_table_generator(z_observations=self.z_obs_unitized, kernel_evals=self.kernel_evals)
        self.p_hat_2d = phat2_table_generator(z_observations=self.z_obs_unitized, kernel_evals=self.kernel_evals)
        self.M, self.d = self.p_hat_1d.shape
        self.maxIter = 400
        self.dx = 1/(self.M-1)


class GridDataPLSBF:
    def __init__(self, W, Z, baseline_kernel, W_groups, bandwidth=None):
        self.M = 100
        self.d = Z.shape[1]
        self.n = W.shape[0]
        self.dz = 1 / (self.M - 1)
        if Z.shape[1] > 0:
            Z_min = np.min(Z, axis=0)
            Z_max = np.max(Z, axis=0)
            self.z_unit_grid = np.linspace(np.zeros(self.d), np.ones(self.d), self.M)
            self.z_grid = np.linspace(Z_min, Z_max, self.M)
            self.z_obs_unitized = (Z - Z_min) / (Z_max - Z_min)
            self.z_observations = Z
            if bandwidth is not None:
                self.bandwidth = np.ones(self.d) * bandwidth
            else:
                self.bandwidth = initialize_h(Z=self.z_obs_unitized)

            # Precompute tables
            self.kernel_evals = Kh_table_generator(self.z_obs_unitized, self.z_unit_grid, baseline_kernel, self.bandwidth)
            self.p_hat_1d = phat_table_generator(self.z_obs_unitized, self.kernel_evals)
            self.p_hat_2d = phat2_table_generator(self.z_obs_unitized, self.kernel_evals)
            self.dx = 1/(self.M-1)

        self.maxIter = 400
        self.m = W.shape[1]
        self.W = W
        self.w_means = W.mean(axis=0)
        self.w_center = W - self.w_means
        self.w_std_dev = self.W.std(axis=0)
        self.w_standardized = standardize(self.W)
        self.index_map, self.L = generate_index_map(W_groups)



def initialize_h(Z):
    """
    We use Silvermans bandwidth.
    """
    n, d = Z.shape
    q3, q1 = np.percentile(Z, [75, 25], axis=0)
    IQR = q3 - q1
    m_vec = np.minimum(np.std(Z, axis=0), IQR / 1.349)
    h = (0.9 / np.power(n, 1 / 5)) * m_vec

    # Check phat(x) > 0 for all x
    sorted_Z = np.sort(Z, axis=0)
    diff_array = 0.5 * np.diff(sorted_Z, axis=0)
    min_h = np.max(diff_array, axis=0)
    return np.maximum(min_h, h)




###################
###################
###################
###################
###################
class GridData:
    def __init__(self, x_observations, x_grid, kernel, bandwidth):
        # Precompute tables
        self.kernel_evals = Kh_table_generator(x_observations, x_grid, kernel, bandwidth)
        self.p_hat_1d = phat_table_generator(x_observations, self.kernel_evals)
        self.p_hat_2d = phat2_table_generator(x_observations, self.kernel_evals)
        self.M, self.d = self.p_hat_1d.shape
        self.maxIter = 400
        self.dx = 1/(self.M-1)

class GridDataMixed:
    def __init__(self, w_observations, z_observations, y, z_grid, kernel, bandwidth):
        # Precompute tables
        self.kernel_evals = Kh_table_generator(z_observations, z_grid, kernel, bandwidth)
        self.p_hat_1d = phat_table_generator(z_observations, self.kernel_evals)
        self.p_hat_2d = phat2_table_generator(z_observations, self.kernel_evals)
        #self.nw_est = nadaraya_watson_estimator(y=y, p_hat=self.p_hat_1d, kernel_evals=self.kernel_evals)
        self.M, self.d = self.p_hat_1d.shape
        self.maxIter = 400
        self.dx = 1/(self.M-1)
        self.maxIter = 400

        self.m = w_observations.shape[1]
        self.x_grid = z_grid
        self.W = w_observations
        self.w_means = w_observations.mean(axis=0)
        self.w_center = self.W - self.w_means
        self.z_observations = z_observations

        """  self.d_d = len(W_list)
        self.d_c = z_observations.shape[1]
        self.Y = y
        self.Z = z_observations
        self.n = n
        self.stdW_list = standardize(W_list)
        self.Z_min = Z_min
        self.Z_max = Z_max"""

class GridDataMixedLasso:
    def __init__(self, w_observations, z_observations, z_grid, kernel, bandwidth, w_groups = None):
        # Precompute tables
        self.kernel_evals = Kh_table_generator(z_observations, z_grid, kernel, bandwidth)
        self.p_hat_1d = phat_table_generator(z_observations, self.kernel_evals)
        self.p_hat_2d = phat2_table_generator(z_observations, self.kernel_evals)
        #self.nw_est = nadaraya_watson_estimator(y=y, p_hat=self.p_hat_1d, kernel_evals=self.kernel_evals)
        self.M, self.d = self.p_hat_1d.shape
        self.dx = 1/(self.M-1)
        self.maxIter = 400

        self.m = w_observations.shape[1]
        self.x_grid = z_grid
        self.W = w_observations
        self.w_means = w_observations.mean(axis=0)
        self.w_std_dev = self.W.std(axis=0)
        self.w_standardized = standardize(self.W)
        self.z_observations = z_observations

        # the w_groups is a list of length L
        if w_groups is None:
            w_groups = [i for i in range(self.m)]
        self.index_map, self.L = generate_index_map(w_groups)
        self.n = w_observations.shape[0]

        """  self.d_d = len(W_list)
        self.d_c = z_observations.shape[1]
        self.Y = y
        self.Z = z_observations
        self.n = n
        self.stdW_list = standardize(W_list)
        self.Z_min = Z_min
        self.Z_max = Z_max"""


def Kh_table_generator(z, z_grid, kernel, h):
    """
    Output is (M, n, d)
    """
    output = kernel_h(z=z_grid[:, None, :] - z, baseline_kernel=kernel, h=h)
    output /= np.trapz(output, z_grid[:, np.newaxis], axis=0)
    return output


def phat_table_generator(z_observations, kernel_evals):
    n = z_observations.shape[0]
    return np.sum(kernel_evals, axis=1) / n


def phat2_table_generator(z_observations, kernel_evals):
    n = z_observations.shape[0]
    output = np.tensordot(kernel_evals, kernel_evals, axes=(1, 1)) / n
    output = np.transpose(output, axes=(0, 2, 1, 3))
    return output


def nadaraya_watson_estimator(y, p_hat, kernel_evals):
    # Output is (M,d)
    M, d = p_hat.shape
    n = y.shape[0]
    p = y.shape[1]
    output = np.zeros((M, d, p))
    for k in range(p):
        output[:,:,k] = np.tensordot(y[:,k], kernel_evals, axes=(0, 1))
        output[:,:,k] /= (n * p_hat)
    return output

def standardize(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

def generate_index_map(ind_list):
    """
    ind_list is a list of length m, where the j'th element indicates which group the j'th linear covariate belongs to.
    Thus, if ind_list = [0, 1, 2, 3, 4] then all coordinates are in distinct groups. On the other hand, if
    ind_list = [0, 0, 1, 2, 3] then the first two linear covariates belong to the same group.
    """
    unique_grp = list(dict.fromkeys(ind_list))
    L = len(unique_grp)
    dictionary = {}
    for l in range(L):
        dictionary[l] = [i for i in range(len(ind_list)) if ind_list[i] == unique_grp[l]]
    return dictionary, L