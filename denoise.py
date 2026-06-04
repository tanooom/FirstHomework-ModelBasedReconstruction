import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

from IPPy import operators, solvers, utilities

# Caricamento immagine ground truth
img_np = rgb2gray(data.astronaut())
img_np = resize(img_np, (256, 256), anti_aliasing=True)

x_gt = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

# Creazione del forward problem
noise_level = 0.2
noise = noise_level * torch.rand_like(x_gt)
y_noisy = x_gt + noise

# CGLS (Tikhonov via Early Stopping)
# Nel Denoising, l'operatore A corrisponde all'Identità
A_id = operators.Identity(img_shape=(256, 256))

# Inizializzo il solver
solver_cgls = solvers.CGLS(K=A_id)

# CGLS è un algoritmo iterativo. Il numero di iterazioni funge da regolarizzatore (Early Stopping):
# poche iterazioni --> immagine liscia, senza rumore (equivale a lambda grande)
# tante iterazioni --> ricostruisce anche il rumore (equivale a lambda piccolo)
n_iterazioni = 100

x_cgls, info_cgls = solver_cgls(y_noisy, x_gt, None, maxiter=n_iterazioni)



# TOTAL VARIATION con ChambollePock
solver_tv = solvers.ChambollePockTpVUnconstrained(K=A_id)
lambdas = [0.01, 0.05, 0.1, 0.3, 0.5]

risultati_tv = []
for lmbda in lambdas:
    print(f"Eseguo TV con λ={lmbda}...")
    x_tv, info_tv = solver_tv(y_noisy, x_gt, None, lmbda=lmbda, maxiter=100)
    risultati_tv.append((lmbda, x_tv, info_tv))



#TOTAL p-Variation (p<1) con ChambollePock
lmbda_tpv = 0.1
p_values = [0.8, 0.6, 0.4, 0.2, 0.1]

risultati_tpv = []
for p in p_values:
    print(f"Eseguo TpV con p={p}, λ={lmbda_tpv}...")
    x_tpv, info_tpv = solver_tv(y_noisy, x_gt, None, lmbda=lmbda_tpv, maxiter=100, p=p)
    risultati_tpv.append((p, x_tpv, info_tpv))


#======================
# GRID SEARCH PER TPV
#======================


p_grid = [0.8, 0.6, 0.4, 0.2]
lambda_grid = [0.01, 0.05, 0.1, 0.3, 0.5]

grid_results = {p: {} for p in p_grid}
for p in p_grid:
    for lmbda in lambda_grid:
        print(f"Grid search TpV: p={p}, λ={lmbda}...")
        x_g, info_g = solver_tv(y_noisy, x_gt, None, lmbda=lmbda, maxiter=100, p=p)
        grid_results[p][lmbda] = (x_g, info_g)

# ════════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONE IMMAGINI
# ════════════════════════════════════════════════════════════════════════════

# Figura 1: Tikhonov
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("Tikhonov (CGLS - Early Stopping)", fontsize=13)

axes1[0].imshow(x_gt.squeeze().cpu().detach().numpy(), cmap='gray')
axes1[0].set_title("Ground Truth")
axes1[0].axis('off')

axes1[1].imshow(y_noisy.squeeze().cpu().detach().numpy(), cmap='gray')
axes1[1].set_title(f"Rumorosa (noise={noise_level})")
axes1[1].axis('off')

axes1[2].imshow(x_cgls.squeeze().cpu().detach().numpy(), cmap='gray')
axes1[2].set_title(f"Tikhonov - CGLS (iter={n_iterazioni})")
axes1[2].axis('off')

plt.tight_layout()

# Figura 2: TV con 5 lambda
fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5))
fig2.suptitle("Total Variation (p=1) - Chambolle-Pock (confronto λ)", fontsize=13)

for i, (lmbda, x_tv, info_tv) in enumerate(risultati_tv):
    axes2[i].imshow(x_tv.squeeze().cpu().detach().numpy(), cmap='gray')
    axes2[i].set_title(f"λ={lmbda}\n(iter={info_tv['iterations']})")
    axes2[i].axis('off')

plt.tight_layout()

# Figura 3: TpV con 5 valori di p (lambda fisso)
fig3, axes3 = plt.subplots(1, 5, figsize=(25, 5))
fig3.suptitle(f"Total p-Variation - Chambolle-Pock (λ={lmbda_tpv}, confronto p)", fontsize=13)

for i, (p, x_tpv, info_tpv) in enumerate(risultati_tpv):
    axes3[i].imshow(x_tpv.squeeze().cpu().detach().numpy(), cmap='gray')
    axes3[i].set_title(f"p={p}\n(iter={info_tpv['iterations']})")
    axes3[i].axis('off')

plt.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# METRICHE VS ITERAZIONI
# ════════════════════════════════════════════════════════════════════════════

# Figura 4: metriche CGLS
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
fig4.suptitle("Metriche vs Iterazioni - Tikhonov (CGLS)", fontsize=13)

axes4[0].plot(info_cgls['PSNR'].squeeze().cpu().detach().numpy())
axes4[0].set_title("PSNR")
axes4[0].set_xlabel("Iterazione")
axes4[0].set_ylabel("dB")
axes4[0].grid(True)

axes4[1].plot(info_cgls['SSIM'].squeeze().cpu().detach().numpy())
axes4[1].set_title("SSIM")
axes4[1].set_xlabel("Iterazione")
axes4[1].grid(True)

axes4[2].plot(info_cgls['RE'].squeeze().cpu().detach().numpy())
axes4[2].set_title("Relative Error")
axes4[2].set_xlabel("Iterazione")
axes4[2].grid(True)

plt.tight_layout()

# Figura 5: metriche TV (confronto lambda)
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4))
fig5.suptitle("Metriche vs Iterazioni - TV (confronto λ)", fontsize=13)

for lmbda, _, info_tv in risultati_tv:
    axes5[0].plot(info_tv['PSNR'].squeeze().cpu().detach().numpy(), label=f"λ={lmbda}")
    axes5[1].plot(info_tv['SSIM'].squeeze().cpu().detach().numpy(), label=f"λ={lmbda}")
    axes5[2].plot(info_tv['RE'].squeeze().cpu().detach().numpy(), label=f"λ={lmbda}")

for ax, title, ylabel in zip(axes5, ["PSNR", "SSIM", "Relative Error"], ["dB", "", ""]):
    ax.set_title(title)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# Figura 6: metriche TpV (confronto p, lambda fisso)
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4))
fig6.suptitle(f"Metriche vs Iterazioni - TpV (λ={lmbda_tpv}, confronto p)", fontsize=13)

for p, _, info_tpv in risultati_tpv:
    axes6[0].plot(info_tpv['PSNR'].squeeze().cpu().detach().numpy(), label=f"p={p}")
    axes6[1].plot(info_tpv['SSIM'].squeeze().cpu().detach().numpy(), label=f"p={p}")
    axes6[2].plot(info_tpv['RE'].squeeze().cpu().detach().numpy(), label=f"p={p}")

for ax, title, ylabel in zip(axes6, ["PSNR", "SSIM", "Relative Error"], ["dB", "", ""]):
    ax.set_title(title)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# GRID SEARCH TpV - Tabella e Heatmap
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("GRID SEARCH TpV - Metriche finali")
print(f"{'p':<6} {'λ':<8} {'PSNR':>10} {'SSIM':>10} {'RE':>10} {'Iter':>6}")
print("="*65)

best_psnr, best_ssim = -np.inf, -np.inf
best_psnr_config, best_ssim_config = None, None

psnr_matrix = np.zeros((len(p_grid), len(lambda_grid)))
ssim_matrix = np.zeros((len(p_grid), len(lambda_grid)))

for i, p in enumerate(p_grid):
    for j, lmbda in enumerate(lambda_grid):
        _, info_g = grid_results[p][lmbda]
        k    = info_g['iterations'] - 1
        psnr = info_g['PSNR'][k].item()
        ssim = info_g['SSIM'][k].item()
        re   = info_g['RE'][k].item()
        itr  = info_g['iterations']

        psnr_matrix[i, j] = psnr
        ssim_matrix[i, j] = ssim

        print(f"{p:<6} {lmbda:<8} {psnr:>10.4f} {ssim:>10.4f} {re:>10.4f} {itr:>6}")

        if psnr > best_psnr:
            best_psnr, best_psnr_config = psnr, (p, lmbda)
        if ssim > best_ssim:
            best_ssim, best_ssim_config = ssim, (p, lmbda)

print("="*65)
print(f"Miglior PSNR: {best_psnr:.4f} dB  → p={best_psnr_config[0]}, λ={best_psnr_config[1]}")
print(f"Miglior SSIM: {best_ssim:.4f}     → p={best_ssim_config[0]}, λ={best_ssim_config[1]}")

# Figura 7: Heatmap grid search
fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))
fig7.suptitle("Grid Search TpV - Heatmap metriche finali", fontsize=13)

for ax, matrix, title, fmt in zip(
    axes7,
    [psnr_matrix, ssim_matrix],
    ["PSNR (dB)", "SSIM"],
    [".2f", ".4f"]
):
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(lambda_grid)))
    ax.set_xticklabels([str(l) for l in lambda_grid])
    ax.set_yticks(range(len(p_grid)))
    ax.set_yticklabels([str(p) for p in p_grid])
    ax.set_xlabel("λ")
    ax.set_ylabel("p")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for ii in range(len(p_grid)):
        for jj in range(len(lambda_grid)):
            ax.text(jj, ii, format(matrix[ii, jj], fmt),
                    ha='center', va='center', color='white', fontsize=9)

plt.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# TABELLA RIASSUNTIVA METRICHE FINALI
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"{'Metodo':<30} {'PSNR':>10} {'SSIM':>10} {'RE':>10} {'Iter':>6}")
print("="*70)

# Rumorosa (baseline)
psnr_noisy = 10 * torch.log10(1 / torch.mean((x_gt - y_noisy)**2)).item()
print(f"{'Rumorosa (baseline)':<30} {psnr_noisy:>10.4f} {'  -':>10} {'  -':>10} {'  -':>6}")

# CGLS
k = info_cgls['iterations'] - 1
print(f"{'Tikhonov CGLS':<30} "
      f"{info_cgls['PSNR'][k].item():>10.4f} "
      f"{info_cgls['SSIM'][k].item():>10.4f} "
      f"{info_cgls['RE'][k].item():>10.4f} "
      f"{info_cgls['iterations']:>6}")

# TV
for lmbda, _, info_tv in risultati_tv:
    k = info_tv['iterations'] - 1
    print(f"{'TV  λ=' + str(lmbda):<30} "
          f"{info_tv['PSNR'][k].item():>10.4f} "
          f"{info_tv['SSIM'][k].item():>10.4f} "
          f"{info_tv['RE'][k].item():>10.4f} "
          f"{info_tv['iterations']:>6}")

# TpV (lambda fisso, confronto p)
for p, _, info_tpv in risultati_tpv:
    k = info_tpv['iterations'] - 1
    print(f"{'TpV p=' + str(p) + ' λ=' + str(lmbda_tpv):<30} "
          f"{info_tpv['PSNR'][k].item():>10.4f} "
          f"{info_tpv['SSIM'][k].item():>10.4f} "
          f"{info_tpv['RE'][k].item():>10.4f} "
          f"{info_tpv['iterations']:>6}")

# TpV best (da grid search)
p_best_psnr, l_best_psnr = best_psnr_config
_, info_best = grid_results[p_best_psnr][l_best_psnr]
k = info_best['iterations'] - 1
print(f"{'TpV best PSNR (grid)':<30} "
      f"{info_best['PSNR'][k].item():>10.4f} "
      f"{info_best['SSIM'][k].item():>10.4f} "
      f"{info_best['RE'][k].item():>10.4f} "
      f"{info_best['iterations']:>6}")

print("="*70)

plt.show()