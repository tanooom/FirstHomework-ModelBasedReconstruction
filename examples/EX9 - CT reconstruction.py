import os
import sys
import time

import numpy as np
import torch

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import operators, utilities, solvers
from IPPy.utilities import data, metrics
from IPPy.utilities import load_image, save_image, normalize
from IPPy.utilities.metrics import PSNR, SSIM, RE


#################################################
### SETTING THINGS UP
#################################################

# Set a seed (for reproducibility)
torch.manual_seed(0)

# Set required parameters
# Load GT image
x_true = load_image("../data/Mayo/test/C081/0.png")

start_angle = 0  # first angle of angular range
end_angle = 180  # last angle of angular range
n_angles = 180  # number of projections
det_size = 512  # detector resolution

#geometry = "fanflat"  # This is the only one available right now
noise_level = 0.000

lmbda = 0.01  # The regularization parameter for the algorithm
maxiter = 100  # Number of maximum iterations for the solver
p = 1  # Sparsity parameter (only for ChambollePock solver)

# Define CTOperator
#angles = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), n_angles)
K = operators.CTProjector(
    img_shape=(512,512),
    angles=np.linspace(0, np.pi, 60),
    det_size=512,
    geometry="parallel",
)

""" K = operators.CTProjector(
    img_shape=(512,512),
    angles=angles,
    geometry=geometry,
    source_origin=2800,
    origin_det=500,
)
 """

# Compute noisy sinogram
y = K(x_true)  # Forward projection
y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)

solver1=solvers.FBP(K)

# SOLUTION
x_sol,info = solver1(
    y_delta,
    x_true=x_true,
    starting_point=y_delta,
)



    # METRICS

    # Compute metrics
psnr = PSNR(x_sol, x_true)
ssim = SSIM(x_sol, x_true)
re = RE(x_sol, x_true)
print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")


save_image(normalize(x_true), "gt_image.png")
save_image(y_delta, "sinogram.png")
save_image(normalize(x_sol), "FBP.png")

# Initialize solver
solver = solvers.ChambollePockTpVUnconstrained(K)

#################################################
### EXECUTION
#################################################


# SOLUTION
x_sol, info = solver(
    y_delta,
    lmbda=lmbda,
    starting_point=None,
    x_true=x_true,
    maxiter=maxiter,
    p=p,
    verbose=True,
)

    # METRICS
print(
    f"RE = {metrics.RE(x_sol, x_true):0.4f}, ",
    f"PSNR = {metrics.PSNR(x_sol, x_true):0.4f}, SSIM = {metrics.SSIM(x_sol, x_true):0.4f}.",
)
save_image(normalize(x_sol), "reconstruction_TV.png")
