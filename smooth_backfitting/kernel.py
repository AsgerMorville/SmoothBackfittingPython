import numpy as np

def kernel_h(z, baseline_kernel, h):
    """
    This function scales the baseline kernel with the chosen bandwidth size
    """
    znew = np.zeros(z.shape)
    d = z.shape[2]
    for j in range(d):
        znew[:, :, j] = z[:, :, j] / h[j]
    return (1 / h) * baseline_kernel(znew)


class EpanKernel:
    def __call__(self, z):
        mask = np.abs(z) < 1
        result = np.zeros_like(z)
        result[mask] = (3 / 4) * (1 - np.power(z[mask], 2))
        return result



