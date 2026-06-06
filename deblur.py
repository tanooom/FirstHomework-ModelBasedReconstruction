import torch
import matplotlib.pyplot as plt
from pathlib import Path

from IPPy import operators, solvers, utilities
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy.utilities import load_image, save_image, normalize

device = utilities.get_device()
print(f"Device used: {device}.")

# Caricamento immagine dal dataset Mayo
x_true = load_image("12_mayo.png")
print(f"Shape GT: {list(x_true.shape)}")

# Lista di lambda da provare per la ricerca del parametro ottimale
lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

# Parametro di sicurezza per il Discrepancy Principle
eta = 1.0

# Due configurazioni: blur leggero vs blur pesante
configurazioni = [
    {"kernel_size": 7,  "sigma": 1.0, "noise_level": 0.01},  # blur leggero, poco rumore
    {"kernel_size": 15, "sigma": 3.0, "noise_level": 0.05},  # blur pesante, piu rumore
]

# ════════════════════════════════════════════════════════════════════════════
# LOOP SULLE CONFIGURAZIONI
# ════════════════════════════════════════════════════════════════════════════
for cfg in configurazioni:

    kernel_size = cfg["kernel_size"]
    sigma       = cfg["sigma"]
    noise_level = cfg["noise_level"]

    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE: kernel={kernel_size}, sigma={sigma}, noise={noise_level}")
    print(f"{'='*60}")

    # Crea cartella risultati specifica per questa configurazione
    results_dir = Path(f"deblur_results/k{kernel_size}_s{sigma}_n{noise_level}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Operatore di blur gaussiano
    # kernel_size e sigma controllano la dimensione e l'intensita del blur
    K = operators.Blurring(
        img_shape=x_true.shape[-2:],
        kernel_type="gaussian",
        kernel_size=kernel_size,
        kernel_variance=sigma**2,
    )

    # Forward problem: blur + rumore gaussiano
    # noise level basso perche il deblur e gia un problema mal posto
    y_delta = K(x_true) + noise_level * torch.randn_like(x_true)
    print(f"Shape misurazioni: {list(y_delta.shape)}")

    save_image(normalize(x_true), results_dir / "gt_image.png")
    save_image(normalize(y_delta), results_dir / "blurred_image.png")

    # ── Ricerca lambda ottimale - BEST METRIC ────────────────────────────────
    # Prova tutti i lambda e tiene quello con PSNR piu alto
    print("\n--- Ricerca lambda ottimale (Best Metric) ---")
    solver_tv = solvers.ChambollePockTpVUnconstrained(K)

    best_psnr   = -float('inf')
    lambda_best = lambda_list[0]

    for lmbda in lambda_list:
        x_tmp, _ = solver_tv(
            y_delta,
            x_true=x_true,
            starting_point=torch.ones_like(x_true),
            lmbda=lmbda,
            maxiter=100,
            tolf=1e-6,
            tolx=1e-6,
            p=1.0,
            verbose=False,
        )
        psnr_tmp = PSNR(x_tmp, x_true)
        print(f"  lambda={lmbda:.0e} -> PSNR: {psnr_tmp:.2f} dB")

        if psnr_tmp > best_psnr:
            best_psnr   = psnr_tmp
            lambda_best = lmbda

    print(f"=> Lambda ottimale (Best Metric): {lambda_best} | PSNR: {best_psnr:.2f} dB")

    # ── Ricerca lambda ottimale - DISCREPANCY PRINCIPLE ──────────────────────
    # Cerca il lambda per cui il residuo e circa uguale al livello di rumore
    # ||K(x_sol) - y_delta|| <= eta * noise_level * ||y_delta||
    print("\n--- Ricerca lambda ottimale (Discrepancy Principle) ---")
    target    = eta * noise_level * torch.norm(y_delta)
    lambda_dp = lambda_list[-1]
    print(f"  Target residuo: {target:.4f}")

    for lmbda in lambda_list:
        x_tmp, _ = solver_tv(
            y_delta,
            x_true=x_true,
            starting_point=torch.ones_like(x_true),
            lmbda=lmbda,
            maxiter=100,
            tolf=1e-6,
            tolx=1e-6,
            p=1.0,
            verbose=False,
        )
        residuo = torch.norm(K(x_tmp) - y_delta)
        print(f"  lambda={lmbda:.0e} -> residuo: {residuo:.4f} (target: {target:.4f})")

        if residuo <= target:
            lambda_dp = lmbda
            break

    print(f"=> Lambda ottimale (Discrepancy Principle): {lambda_dp}")

    # ════════════════════════════════════════════════════════════════════════
    # TIKHONOV via CGLS
    # ════════════════════════════════════════════════════════════════════════
    # Nel deblur il semi-convergence e ancora piu marcato del denoising:
    # inizialmente CGLS migliora la ricostruzione, poi il rumore amplificato
    # dall'inversione del blur degrada rapidamente la soluzione
    print("\n--- CGLS Tikhonov ---")
    solver_cgls = solvers.CGLS(K)

    x_cgls, info_cgls = solver_cgls(
        y_delta,
        x_true=x_true,
        starting_point=torch.zeros_like(x_true),
        lam=lambda_best,
        maxiter=100,
        tolf=1e-7,
        tolx=1e-7,
        verbose=True,
    )

    psnr = PSNR(x_cgls, x_true)
    ssim = SSIM(x_cgls, x_true)
    re   = RE(x_cgls, x_true)
    print(f"Tikhonov CGLS -> PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")
    save_image(normalize(x_cgls), results_dir / "deblurred_cgls.png")

    # ════════════════════════════════════════════════════════════════════════
    # TV con lambda_best (Best Metric)
    # ════════════════════════════════════════════════════════════════════════
    # TV e particolarmente efficace nel deblur perche preserva i bordi
    # che il blur tende ad ammorbidire
    print("\n--- TV (Best Metric) ---")
    x_tv_best, info_tv_best = solver_tv(
        y_delta,
        x_true=x_true,
        starting_point=torch.ones_like(x_true),
        lmbda=lambda_best,
        maxiter=100,
        tolf=1e-6,
        tolx=1e-6,
        p=1.0,
        verbose=True,
    )

    psnr = PSNR(x_tv_best, x_true)
    ssim = SSIM(x_tv_best, x_true)
    re   = RE(x_tv_best, x_true)
    print(f"TV (Best Metric, lambda={lambda_best}) -> PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")
    save_image(normalize(x_tv_best), results_dir / "deblurred_tv_best.png")

    # ════════════════════════════════════════════════════════════════════════
    # TV con lambda_dp (Discrepancy Principle)
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- TV (Discrepancy Principle) ---")
    x_tv_dp, info_tv_dp = solver_tv(
        y_delta,
        x_true=x_true,
        starting_point=torch.ones_like(x_true),
        lmbda=lambda_dp,
        maxiter=100,
        tolf=1e-6,
        tolx=1e-6,
        p=1.0,
        verbose=True,
    )

    psnr = PSNR(x_tv_dp, x_true)
    ssim = SSIM(x_tv_dp, x_true)
    re   = RE(x_tv_dp, x_true)
    print(f"TV (Discrepancy Principle, lambda={lambda_dp}) -> PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")
    save_image(normalize(x_tv_dp), results_dir / "deblurred_tv_dp.png")

    # ════════════════════════════════════════════════════════════════════════
    # TpV con lambda_best
    # ════════════════════════════════════════════════════════════════════════
    # Con p<1 TpV tende a recuperare dettagli fini che TV tende a smussare
    print("\n--- TpV (Best Metric) ---")
    solver_tpv = solvers.ChambollePockTpVUnconstrained(K)
    p = 0.5

    x_tpv, info_tpv = solver_tpv(
        y_delta,
        x_true=x_true,
        starting_point=torch.ones_like(x_true),
        lmbda=lambda_best,
        maxiter=100,
        tolf=1e-6,
        tolx=1e-6,
        p=p,
        verbose=True,
    )

    psnr = PSNR(x_tpv, x_true)
    ssim = SSIM(x_tpv, x_true)
    re   = RE(x_tpv, x_true)
    print(f"TpV (p={p}, lambda={lambda_best}) -> PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | RE: {re:.4f}")
    save_image(normalize(x_tpv), results_dir / "deblurred_tpv.png")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT METRICHE VS ITERAZIONI
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(f"Metriche vs Iterazioni - Deblur (k={kernel_size}, s={sigma}, noise={noise_level})", fontsize=14)

    metodi = [
        ("Tikhonov CGLS",              info_cgls),
        (f"TV Best (λ={lambda_best})",  info_tv_best),
        (f"TV DP (λ={lambda_dp})",      info_tv_dp),
        (f"TpV (p={p})",               info_tpv),
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

    print(f"\nRisultati salvati in: {results_dir}")