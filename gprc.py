import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

def configure_gprc(use_gprc, seed):

    """
    Configure the GPRC model for residual correction.

    Parameters
    ----------
    use_gprc: bool
        Enable or disable GPRC.
    seed: int
        Random seed.

    Returns
    -------
    gp : GaussianProcessRegressor
        Gaussian Process model.
    scaler: MinMaxScaler
        Fitted scaler for normalizing GPRC input parameters.
    """
        
    if use_gprc == False:
        return None, None

    data = np.loadtxt("data/gprc_training_data.csv", delimiter=",", skiprows=1)

    x = data[:,2:5]
    y_real = data[:,0]
    y_sim = data[:,1]

    y_residuals = y_real - y_sim

    scaler = MinMaxScaler()
    scaler.fit(np.array([[8, 0.8, 1], [18, 3, 2]])) # Lower and upper limits of parameters
    x_scaled = scaler.transform(x)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale = [1.0, 1.0, 1.0], length_scale_bounds = (1e-2, 1e2)) + WhiteKernel(noise_level = 1.0, noise_level_bounds = (1e-2, 20.0))

    gp = GaussianProcessRegressor(
        kernel = kernel,
        n_restarts_optimizer = 10,
        normalize_y = True,
        random_state = seed
    )
    gp.fit(x_scaled, y_residuals)

    return gp, scaler