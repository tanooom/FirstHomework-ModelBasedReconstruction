import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize


from IPPy import operators, solvers, utilities

#caricamento immagine ground truth
#utilizzo immagine astronaut di skimage convertendola in scala di grigi
img_np = rgb2gray(data.astronaut())

#effettuo ridimensionamento per rendere più veloci i calcoli
img_np = resize(img_np, (256,256), anti_aliasing=True)

x_gt = torch.tensor(img_np, dtype=torch.float32)

#creazione del forward problem
#parametro che possiamo far variare
noise_level = 0.2 
noise = noise_level * torch.rand_like(x_gt)

y_noisy = x_gt + noise

#Visualizzazione
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(x_gt.numpy(), cmap='gray')
plt.title("Ground Truth (x)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(y_noisy.numpy(), cmap='gray')
plt.title(f"Misurazione Rumorosa (y) | Livello: {noise_level}")
plt.axis('off')

plt.tight_layout()
plt.show()


