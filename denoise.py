import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path

from IPPy import operators, solvers, utilities
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy.utilities import load_image, save_image, normalize

results_dir = Path("denoise_results")

device = utilities.get_device()
print(f"Device used: {device}.")

#carico immagine scaricata dal dataset mayo
#load image mi torna già tensore (1,1,H,W) float 32 pronto per i solver
x_true = load_image("12_mayo.png")
print(f"Shape GT: {list(x_true.shape)}")


#Creazione Operatore Identità
K = operators.Identity(
    img_shape=x_true.shape[-2:],
)

#Forward Problem
#Operatore K è Identità: nel denoising non c'è degradazione spaziale
# noise level controlla intensità del rumore gaussiano che viene aggiunto

noise_level = 0.05
y_delta = K(x_true) + noise_level *torch.randn_like(x_true)
print(f"Shape misurazioni: {list(y_delta.shape)}")

#salvataggio immagini di input
#normalize() riporta i valori nel range [0,1] prima di salvare,
# necessario perché il rumore gaussiano non può portare numeri fuori range
save_image(normalize(x_true), results_dir / "gt_image.png")
save_image(normalize(y_delta), results_dir / "noised_image.png")


# ---- CGLS - Tikhonov -----
# risolve problema ai minimi quadrati con regolarizzazione implicita tramite early stopping
#più iterazioni, meno regolarizzazione

lambda_tik = 0.1
maxiter = 100
tolf = tolx = 1e-7

solver_cgls = solvers.CGLS(K)

x_cgls, info_cgls = solver_cgls(
    y_delta,
    x_true = x_true,
    starting_point=torch.zeros_like(x_true),
    lam = lambda_tik,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    verbose=True,
)

#Metriche
psnr = PSNR(x_cgls, x_true)
ssim = SSIM(x_cgls, x_true)
re = RE(x_cgls, x_true)

print(f"Tikhonov CGLS --> PSNR: {psnr: .2f} db | SSIM: {ssim: .4f} | RE: {re: .4f}")

save_image(normalize(x_cgls), results_dir / "denoised_cgls.png")

# ---- Total Variation con p = 1 -----
# Minimizza la total variation dell'immagine

#lmbda controlla il trade.off: grande = più smoothing, piccolo = più fedeltà ai dati
#tolf e tolx fermano algoritmo anticipatamente se la soluzione è già conversa

lambda_tv = 5e-4
maxiter = 200
tolf = tolx = 1e-6

solver_tv = solvers.ChambollePockTpVUnconstrained(K)

x_tv, info_tv = solver_tv(
    y_delta,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
    lmbda=lambda_tv,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    p=1.0,        #corrisponde alla TV classica
    verbose=True,
)

psnr = PSNR(x_tv, x_true)
ssim = SSIM(x_tv, x_true)
re   = RE(x_tv, x_true)
print(f"TV → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_tv), results_dir / "denoised_tv.png")


# ---- Total p Variation con p < 1 -----
# stesso solver di TV, ma con p<1 rendiamo la penalizzazione non convessa
# p<1 permette soluzioni con bordi più netti

#il problema diventa più difficile da ottimizzare
lambda_tpv = 5e-4
maxiter = 200
tolf = tolx = 1e-6
p = 0.5

solver_tpv = solvers.ChambollePockTpVUnconstrained(K)

x_tpv, info_tpv = solver_tpv(
    y_delta,
    x_true=x_true,
    starting_point=torch.zeros_like(x_true),
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

save_image(normalize(x_tpv), results_dir / "denoised_tpv.png")

# --- Plot metriche vs iterazioni ---
# Mostra come PSNR, SSIM e RE evolvono durante le iterazioni
# utile per vedere la convergenza e il fenomeno del semi-convergence in CGLS

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Metriche vs Iterazioni - Denoising", fontsize=14)

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
plt.savefig(results_dir / "metrics_denoising.png", dpi=150, bbox_inches='tight')
plt.show()