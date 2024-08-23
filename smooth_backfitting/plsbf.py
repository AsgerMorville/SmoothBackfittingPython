import numpy as np
from smooth_backfitting.core import plsbf_lasso_loop_sep, plsbf_lasso_loop, plsbf_loop, linear_regression_group_lasso, smooth_backfitting_lasso_loop, linear_regression, smooth_backfitting_loop
from smooth_backfitting.grid_data import GridDataPLSBF, GridDataSBF
from smooth_backfitting.additive_function import PLAdditiveFunction, AdditiveFunction
from smooth_backfitting.kernel import EpanKernel

class BasePLSBF:
    def __init__(self, kernel="epan"):
        if kernel == "epan":
            self.baseline_kernel = EpanKernel()
        else:
            raise ValueError("Unknown baseline kernel function")

    def fit_bw_cv(self, W, Z, Y, W_groups=None):
        grid_length = 10
        min_bandwidth = minimum_bw(Z)
        bandwidth_grid = np.geomspace(min_bandwidth, 1, grid_length)
        prediction_error = np.zeros(grid_length)
        for l in range(grid_length):
            self.fit(W, Z, Y, W_groups, bandwidth_grid[l])
            fitted_y = self.predict(W,Z)
            prediction_error[l] = np.sum(np.power(fitted_y-Y,2))
        best_bandwidth = bandwidth_grid[np.argmin(prediction_error)]
        self.fit(W, Z, Y, W_groups, best_bandwidth)

    def fit(self, W, Z, Y, W_groups=None, h=None):
        # initialize W_groups if none
        if W_groups == None:
            W_groups = [i for i in range(W.shape[1])]

        if W.shape[1] == 0 and Z.shape[1] == 0:
            # No predictors present
            self.additive_function = self.fit_without_W_and_Z(Y)
        elif W.shape[1] == 0 and Z.shape[1] > 0:
            # No linear covariates present, but non-linear covariates present.
            self.additive_function = self.fit_without_W(Z, Y, h)
        elif W.shape[1] > 0 and Z.shape[1] == 0:
            # Linear covariates present, but no non-linear covariates present.
            self.additive_function = self.fit_without_Z(W, Y, W_groups)
        else:
            # Both linear and non-linear covariates present.
            self.additive_function = self.fit_full(W, Z, Y, W_groups, h)

    def fit_without_W_and_Z(self, Y):
        y_mean = Y.mean(axis=0)
        x_grid_points = np.zeros((0, 0))
        m_points = np.zeros((0, 0, Y.shape[1]))
        beta = np.zeros((0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=x_grid_points, m_points=m_points, beta=beta, y_mean=y_mean)

    def fit_without_Z(self, W, Y, W_groups):
        pass

    def fit_without_W(self, Z, Y):
        pass

    def fit_full(self, W, Z, Y, W_groups):
        pass

    def predict(self, W, Z):
        return self.additive_function.predict(Z=Z, W=W)





class PLSBF(BasePLSBF):
    def __init__(self, kernel="epan"):
        super().__init__(kernel)

    def fit_without_W(self, Z, Y, h):
        # This case reduces to regular smooth backfitting with Z=X.
        Y_mean = Y.mean(axis=0)
        grid_data = GridDataSBF(Z=Z, baseline_kernel=self.baseline_kernel, bandwidth=h)
        m_hat = smooth_backfitting_loop(y_center=Y - Y_mean, grid=grid_data)
        beta_hat = np.zeros(shape=(0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=Y_mean)

    def fit_without_Z(self, W, Y, W_groups):
        # This case reduces to linear regression with W=X.
        grid_data = GridDataPLSBF(Z=np.zeros((0,0)), W=W, baseline_kernel=self.baseline_kernel, W_groups=W_groups)
        beta_hat = linear_regression(W=grid_data.w_center, Y=Y)
        z_grid = np.zeros((0, 0))
        m_points = np.zeros((0, 0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=z_grid, m_points=m_points, beta=beta_hat, y_mean=Y.mean(axis=0)-beta_hat.T @ grid_data.w_means)

    def fit_full(self, W, Z, Y, W_groups, h):
        y_mean = Y.mean(axis=0)
        grid_data = GridDataPLSBF(W=W, Z=Z, baseline_kernel=self.baseline_kernel, W_groups=W_groups, bandwidth=h)
        m_hat, beta_hat = plsbf_loop(y_center=Y - y_mean, grid=grid_data)
        # adjust the intercept to account for the centralization of W
        intercept = y_mean-beta_hat.T @ grid_data.w_means
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=intercept)


class PLSBFLasso(BasePLSBF):
    def __init__(self, lmbda=0, kernel="epan"):
        super().__init__(kernel)
        self.lmbda = lmbda

    def fit_without_W(self, Z, Y, h=None):
        # This case reduces to regular smooth backfitting with Z=X.
        Y_mean = Y.mean(axis=0)
        grid_data = GridDataSBF(Z=Z, baseline_kernel=self.baseline_kernel, bandwidth=h)
        m_hat = smooth_backfitting_lasso_loop(y_center=Y - Y_mean, grid=grid_data, lmbda=self.lmbda)
        beta_hat = np.zeros(shape=(0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=Y_mean)

    def fit_without_Z(self, W, Y, W_groups):
        # This case reduces to linear regression with W=X.
        grid_data = GridDataPLSBF(W=W, Z=np.zeros((0,0)), baseline_kernel=self.baseline_kernel, W_groups=W_groups)
        beta_hat = linear_regression_group_lasso(grid=grid_data, Y=Y, lmbda=self.lmbda)
        z_grid_points = np.zeros((0, 0))
        m_points = np.zeros((0, 0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=z_grid_points, m_points=m_points, beta=beta_hat, y_mean=Y.mean(axis=0))

    def fit_full(self, W, Z, Y, W_groups, h=None):
        y_mean = Y.mean(axis=0)
        grid_data = GridDataPLSBF(W=W, Z=Z, baseline_kernel=self.baseline_kernel, W_groups=W_groups, bandwidth=h)
        m_hat, beta_hat = plsbf_lasso_loop(y_center=Y - y_mean, grid=grid_data, lmbda=self.lmbda)
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=y_mean)

class PLSBFLassoSep(BasePLSBF):
    def __init__(self, lmbdaW=0, lmbdaZ=0, kernel="epan"):
        super().__init__(kernel)
        self.lmbdaW = lmbdaW
        self.lmbdaZ = lmbdaZ

    def fit_without_W(self, Z, Y):
        # This case reduces to regular smooth backfitting with Z=X.
        Y_mean = Y.mean(axis=0)
        grid_data = GridDataSBF(Z=Z, baseline_kernel=self.baseline_kernel)
        m_hat = smooth_backfitting_lasso_loop(y_center=Y - Y_mean, grid=grid_data, lmbda=self.lmbdaZ)
        beta_hat = np.zeros(shape=(0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=Y_mean)

    def fit_without_Z(self, W, Y, W_groups):
        # This case reduces to linear regression with W=X.
        grid_data = GridDataPLSBF(W=W, Z=np.zeros((0,0)), baseline_kernel=self.baseline_kernel, W_groups=W_groups)
        beta_hat = linear_regression_group_lasso(grid=grid_data, Y=Y, lmbda=self.lmbdaW)
        z_grid_points = np.zeros((0, 0))
        m_points = np.zeros((0, 0, Y.shape[1]))
        return PLAdditiveFunction(z_grid_points=z_grid_points, m_points=m_points, beta=beta_hat, y_mean=Y.mean(axis=0))

    def fit_full(self, W, Z, Y, W_groups):
        y_mean = Y.mean(axis=0)
        grid_data = GridDataPLSBF(W=W, Z=Z, baseline_kernel=self.baseline_kernel, W_groups=W_groups)
        m_hat, beta_hat = plsbf_lasso_loop_sep(y_center=Y - y_mean, grid=grid_data,
                                               lmbdaW=self.lmbdaW, lmbdaZ=self.lmbdaZ)
        return PLAdditiveFunction(z_grid_points=grid_data.z_grid, m_points=m_hat, beta=beta_hat, y_mean=y_mean)

def minimum_bw(Z):
    Z_min = np.min(Z, axis=0)
    Z_max = np.max(Z, axis=0)
    z_obs_unitized = (Z - Z_min) / (Z_max - Z_min)
    sorted_Z = np.sort(z_obs_unitized, axis=0)
    diff_array = 0.5 * np.diff(sorted_Z, axis=0)
    return np.max(diff_array, axis=0)
