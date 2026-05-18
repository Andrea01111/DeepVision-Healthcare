# ============================================================
#  PROJET SM604 — Partie 1 : MNIST
#  Étape 1.1 : Chargement et préparation des données
# ============================================================
# Ce fichier couvre :
#   - Import et chargement de MNIST via torchvision (données brutes)
#   - Aplatissement des images 28x28 → vecteurs de taille 784
#   - Normalisation des pixels dans [0, 1]
#   - Séparation train / test
#   - Visualisation de quelques exemples
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import os

# ── 1. Téléchargement du jeu de données ─────────────────────
# On utilise torchvision UNIQUEMENT pour télécharger les fichiers bruts.
# On n'utilisera PAS les DataLoader de PyTorch : on veut garder le contrôle sur les données sous forme de tableaux NumPy.

data_dir = "./data_mnist"

train_raw = datasets.MNIST(root=data_dir, train=True,  download=True)
test_raw  = datasets.MNIST(root=data_dir, train=False, download=True)

# ── 2. Conversion en tableaux NumPy ─────────────────────────
# .data   → tenseur uint8  (N, 28, 28), valeurs entières 0–255
# .targets → tenseur int64 (N,)

X_train_img = train_raw.data.numpy()      # shape (60000, 28, 28)
y_train      = train_raw.targets.numpy()  # shape (60000,)
X_test_img   = test_raw.data.numpy()      # shape (10000, 28, 28)
y_test       = test_raw.targets.numpy()   # shape (10000,)

print(f"Train : {X_train_img.shape}, labels : {y_train.shape}")
print(f"Test  : {X_test_img.shape},  labels : {y_test.shape}")
# > Train : (60000, 28, 28), labels : (60000,)
# > Test  : (10000, 28, 28), labels : (10000,)

# ── 3. Aplatissement  28×28 → 784 ───────────────────────────
# Chaque image devient un vecteur ligne xi ∈ R^784.
# On obtient une matrice X de shape (n, 784).

n_train = X_train_img.shape[0]
n_test  = X_test_img.shape[0]

X_train_flat = X_train_img.reshape(n_train, -1)   # (60000, 784)
X_test_flat  = X_test_img.reshape(n_test,  -1)    # (10000, 784)

# ── 4. Normalisation dans [0, 1] ────────────────────────────
# Les pixels sont des entiers entre 0 et 255.
# On divise par 255 pour ramener dans [0, 1].
# Cela accélère la convergence de la descente de gradient
# (les gradients ne sont pas écrasés par de grandes valeurs).

X_train = X_train_flat.astype(np.float64) / 255.0
X_test  = X_test_flat.astype(np.float64)  / 255.0

print(f"\nAprès normalisation :")
print(f"  min={X_train.min():.2f}, max={X_train.max():.2f}, dtype={X_train.dtype}")
# > min=0.00, max=1.00, dtype=float64

# ── 5. Vérification de la distribution des classes ──────────
classes, counts = np.unique(y_train, return_counts=True)
print("\nDistribution des classes (train) :")
for c, n in zip(classes, counts):
    print(f"  Chiffre {c} : {n} exemples")

# ── 6. Visualisation de quelques images ─────────────────────
# On affiche une grille 2×5 : un exemple par classe (0→9).

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Partie 1 — MNIST : un exemple par classe", fontsize=13)

for digit in range(10):
    idx = np.where(y_train == digit)[0][0]          # premier exemple du chiffre
    image = X_train[idx].reshape(28, 28)            # on re-shape pour afficher
    ax = axes[digit // 5][digit % 5]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Classe {digit}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_exemples.png", dpi=100)
plt.show()
print("\nFigure sauvegardée : mnist_exemples.png")

# ── 7. Résumé des dimensions utiles ─────────────────────────
print("\n=== Résumé des dimensions ===")
print(f"X_train : {X_train.shape}  (n_train × 784)")
print(f"y_train : {y_train.shape}  (entiers 0–9)")
print(f"X_test  : {X_test.shape}   (n_test  × 784)")
print(f"y_test  : {y_test.shape}   (entiers 0–9)")

# ── 8. Sauvegarde (optionnelle) ──────────────────────────────
# Utile pour ne pas re-télécharger à chaque session.
os.makedirs("./Partie 1/saves", exist_ok=True)
np.save("./Partie 1/saves/X_train.npy", X_train)
np.save("./Partie 1/saves/y_train.npy", y_train)
np.save("./Partie 1/saves/X_test.npy",  X_test)
np.save("./Partie 1/saves/y_test.npy",  y_test)
print("\nDonnées sauvegardées dans ./Partie 1/saves/")

# ── Pour recharger plus tard ─────────────────────────────────
# X_train = np.load("./Partie 1/saves/X_train.npy")
# y_train = np.load("./Partie 1/saves/y_train.npy")
# X_test  = np.load("./Partie 1/saves/X_test.npy")
# y_test  = np.load("./Partie 1/saves/y_test.npy")