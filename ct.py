import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from IPPy import operators, solvers, utilities
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy.utilities import load_image, save_image, normalize

# Riproducibilità dei risultati
torch.manual_seed(0)

# Crea cartella risultati
results_dir = Path("ct_results")
results_dir.mkdir(exist_ok=True)

# Caricamento immagine dal dataset Mayo
x_true = load_image("12_mayo.png")
print(f"Shape GT: {list(x_true.shape)}")

# Parametri CT
start_angle = 0    # primo angolo del range angolare
end_angle = 180    # ultimo angolo del range angolare
n_angles = 60      # numero di proiezioni: meno angoli = problema più difficile
det_size = 512     # risoluzione del detector

noise_level = 0.0  # nessun rumore di default, il problema è già mal posto
lmbda = 0.01       # parametro di regolarizzazione per Chambolle-Pock
maxiter = 100      # numero massimo di iterazioni
p = 1              # parametro di sparsità (solo per ChambollePock)

# Operatore CT con geometria parallela
# angles: angoli di acquisizione uniformemente distribuiti tra 0 e pi
# det_size: numero di pixel del detector
# geometry: parallel = raggi paralleli
K = operators.CTProjector(
    img_shape=x_true.shape[-2:],
    angles=np.linspace(0, np.pi, n_angles),
    det_size=det_size,
    geometry="parallel",
)

# Forward problem: proiezione CT + rumore gaussiano
# y_delta è il sinogramma: ogni riga è una proiezione a un angolo diverso
# ha shape completamente diversa da x_true
y = K(x_true)
y_delta = y + utilities.gaussian_noise(y, noise_level=noise_level)
print(f"Shape GT:         {list(x_true.shape)}")
print(f"Shape sinogramma: {list(y_delta.shape)}")

# Salvataggio immagini di input
save_image(normalize(x_true), results_dir / "gt_image.png")
save_image(y_delta,           results_dir / "sinogram.png")

# ---- FBP - Filtered Back Projection ----
# Metodo classico analitico, non iterativo
# Veloce ma molto sensibile al rumore e al numero di angoli:
# con pochi angoli produce artefatti a stella (streaking artifacts)
# Funge da baseline per i metodi iterativi
solver_fbp = solvers.FBP(K)

x_fbp, info_fbp = solver_fbp(
    y_delta,
    x_true=x_true,
    starting_point=y_delta,
)

psnr = PSNR(x_fbp, x_true)
ssim = SSIM(x_fbp, x_true)
re   = RE(x_fbp, x_true)
print(f"FBP → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_fbp), results_dir / "ct_fbp.png")

# ---- CGLS - Tikhonov ----
# A differenza di FBP, CGLS risolve iterativamente il sistema lineare
# Il semi-convergence è marcato: con pochi angoli il problema è
# altamente sottoderminato e il rumore viene amplificato rapidamente
lambda_tik = 0.1
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

save_image(normalize(x_cgls), results_dir / "ct_cgls.png")

# ---- TV - Total Variation (p=1) ----
# TV è lo standard nella CT con angoli sparsi
# Le immagini CT hanno strutture con bordi netti (organi, tessuti)
# che TV preserva molto bene grazie alla penalizzazione del gradiente
tolf = tolx = 1e-6

solver_cp = solvers.ChambollePockTpVUnconstrained(K)

x_tv, info_tv = solver_cp(
    y_delta,
    x_true=x_true,
    starting_point=None,
    lmbda=lmbda,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    p=1,
    verbose=True,
)

psnr = PSNR(x_tv, x_true)
ssim = SSIM(x_tv, x_true)
re   = RE(x_tv, x_true)
print(f"TV (p=1) → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_tv), results_dir / "ct_tv.png")

# ---- TpV - Total p-Variation (p<1) ----
# Con p<1 la penalizzazione è più aggressiva nel promuovere la sparsità
# del gradiente — utile in CT dove le strutture anatomiche hanno
# bordi molto netti e zone omogenee ampie
p_tpv = 0.5

x_tpv, info_tpv = solver_cp(
    y_delta,
    x_true=x_true,
    starting_point=None,
    lmbda=lmbda,
    maxiter=maxiter,
    tolf=tolf,
    tolx=tolx,
    p=p_tpv,
    verbose=True,
)

psnr = PSNR(x_tpv, x_true)
ssim = SSIM(x_tpv, x_true)
re   = RE(x_tpv, x_true)
print(f"TpV (p={p_tpv}) → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")

save_image(normalize(x_tpv), results_dir / "ct_tpv.png")

# ---- Plot metriche vs iterazioni ----
# FBP non è iterativo quindi non compare nel plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Metriche vs Iterazioni - CT Reconstruction", fontsize=14)

metodi = [
    ("Tikhonov CGLS",    info_cgls),
    ("TV (p=1)",         info_tv),
    (f"TpV (p={p_tpv})", info_tpv),
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
plt.savefig(results_dir / "metrics_ct.png", dpi=150, bbox_inches='tight')
plt.show()