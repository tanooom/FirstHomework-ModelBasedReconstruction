import torch 
import matplotlib.pyplot as plt
from pathlib import Path

from IPPy import operators, solvers, utilities
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy.utilities import load_image, save_image, normalize

results_dir = Path("deblur_results")

#Caricamento immagine
x_true = load_image("12_mayo.png")
print(f"Shape GT: {list(x_true.shape)}")

#Operatore di blur gaussiano con kernel size=11 e sigma=1.5
kernel_size = 11
sigma = 1.5

K = operators.Blurring(
    img_shape=x_true.shape[-2:],
    kernel_type="gaussian",
    kernel_size=kernel_size,
    kernel_variance=sigma**2,
)

#Forward Problem: blur + rumore gaussiano
#metto noise level basso in quanto deblur è di per se un problema già mal posto
noise_level = 0.01
y_delta = K(x_true) + noise_level * torch.randn_like(x_true)
print(f"Shape misurazioni: {list(y_delta.shape)}")

#salvo immagini di input
save_image(normalize(x_true), results_dir / "gt_image.png")
save_image(normalize(y_delta), results_dir / "blurred_image.png")

# ---- CGLS - Tikhonov -----
# Nel deblur il semi-convergence è ancora più marcato del denoise:
# inizialmente CGLS migliora la ricostruzione, poi il rumore amplificato 
# dall'inversione del blur degrada rapidamente la soluzione
lambda_tik = 0.1
maxiter = 100
tolf = tolx = 1e-7

solver_cgls = solvers.CGLS(K)

x_cgls, info_cgls = solver_cgls(
    y_delta,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
    lam=lambda_tik,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    verbose=True,
)

psnr = PSNR(x_cgls, x_true)
ssim = SSIM(x_cgls, x_true)
re   = RE(x_cgls, x_true)
print(f"Tikhonov CGLS → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_cgls), results_dir / "deblurred_cgls.png")

# ----- TV - Total Variation (p=1) ------
# Tv è particolarmente efficace nel deblur perché preserva i bordi che il blur tende ad ammorbidire

lambda_tv = 1e-2
maxiter = 100
tolf = tolx = 1e-6

solver_tv = solvers.ChambollePockTpVUnconstrained(K)

x_tv, info_tv = solver_tv(
    y_delta,
    x_true=x_true,
    starting_point=torch.ones_like(x_true),
    lmbda=lambda_tv,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    p=1.0,
    verbose=True,
)

psnr = PSNR(x_tv, x_true)
ssim = SSIM(x_tv, x_true)
re   = RE(x_tv, x_true)
print(f"TV → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_tv), results_dir / "deblurred_tv.png")

# ---- TPV - TOtal p Variation con p<1 -----
# con il deblur tpv tende a recuperare dettagli fini che TV tende a smussare

lambda_tpv = 1e-2
maxiter = 100
tolf = tolx = 1e-6
p = 0.5

solver_tpv = solvers.ChambollePockTpVUnconstrained(K)

x_tpv, info_tpv = solver_tpv(
    y_delta,
    x_true=x_true,
    starting_point=torch.ones_like(x_true),
    lmbda=lambda_tpv,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    p=p,
    verbose=True,
)

psnr = PSNR(x_tpv, x_true)
ssim = SSIM(x_tpv, x_true)
re   = RE(x_tpv, x_true)
print(f"TpV (p={p}) → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_tpv), results_dir / "deblurred_tpv.png")

# --- Plot metriche vs iterazioni ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Metriche vs Iterazioni - Deblur", fontsize=14)

metodi = [
    ("Tikhonov CGLS", info_cgls),
    ("TV (p=1)",      info_tv),
    (f"TpV (p={p})",  info_tpv),
]

for col, (nome, info) in enumerate(metodi):
    axes[0, col].plot(info['PSNR'].squeeze().cpu().detach().numpy())
    axes[0, col].set_title(nome)
    axes[0, col].set_ylabel("PSNR (dB)")
    axes[0, col].grid(True)

    axes[1, col].plot(info['SSIM'].squeeze().cpu().detach().numpy())
    axes[1, col].set_ylabel("SSIM")
    axes[1, col].grid(True)

    axes[2, col].plot(info['RE'].squeeze().cpu().detach().numpy())
    axes[2, col].set_ylabel("Relative Error")
    axes[2, col].set_xlabel("Iterazione")
    axes[2, col].grid(True)

plt.tight_layout()
plt.savefig(results_dir / "metrics_deblur.png", dpi=150, bbox_inches='tight')
plt.show()