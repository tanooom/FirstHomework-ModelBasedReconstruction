import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory of 'examples/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPPy import utilities, operators, solvers
from IPPy.utilities import load_image, save_image, normalize
from IPPy.utilities.metrics import PSNR, SSIM, RE

# Set device
device = utilities.get_device()
print(f"Device used: {device}.")

# Load GT image
x_true = load_image("../data/Mayo/test/C081/0.png")
print(f"Shape of the GT: {list(x_true.shape)}.")

# Create the downscaling operator

K = operators.DownScaling(
    img_shape=x_true.shape[-2:],
    downscale_factor=2,
    mode='avg',
)

# Build test problem
noise_level = 0.01
y=K(x_true)
y_delta = y + noise_level * torch.randn_like(y)
print(f"Shape of the measurements: {list(y_delta.shape)}.")

""" print(info.keys())

plt.figure()
plt.plot(info["SSIM"])
plt.xlabel("iterations")
plt.ylabel("SSIM")
plt.show()

# Compute metrics
psnr = PSNR(x_sol, x_true)
ssim = SSIM(x_sol, x_true)
re = RE(x_sol, x_true)
print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")
"""
save_image(normalize(x_true), "gt_image.png")
save_image(normalize(y_delta), "DR_image.png")
#save_image(normalize(x_sol), "deblurred_image_tik.png")

# Set up the solver
lambda_tv = 1e-1
max_iters = 100
solver = solvers.ChambollePockTpVUnconstrained(K)

# Run the solver
x_sol, info = solver(
    y_delta,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
    lmbda=lambda_tv,
    max_iters=max_iters,
    p=1.0,
    verbose=True,
)

# Compute metrics
psnr = PSNR(x_sol, x_true)
ssim = SSIM(x_sol, x_true)
re = RE(x_sol, x_true)
print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

# Save the results
save_image(normalize(x_sol), "SR_image_tv.png")