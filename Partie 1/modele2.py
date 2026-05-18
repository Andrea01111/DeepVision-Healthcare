"""
Projet SM604 - Partie 1.2.2 : Réseau de neurones multi-couches sur MNIST
-------------------------------------------------------------------------
Implémentation from scratch :
  - Réseau avec H = 1 ou H = 2 couches cachées
  - Activation ReLU (couches cachées) + Softmax (sortie)
  - Perte cross-entropy
  - Rétropropagation (backpropagation) complète et générique
  - SGD mini-batch
  - Comparaison avec le modèle linéaire (partie 1.2.1)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# 1. Chargement et préparation des données
# ─────────────────────────────────────────────



print("Chargement des données MNIST depuis ./saves...")

# On vérifie la présence des fichiers générés par mnist_model.py
if not os.path.exists("./Partie 1/saves/X_train.npy"):
    raise FileNotFoundError("Erreur : Lance d'abord 'mnist_model.py' pour générer les fichiers dans ./saves")
os.makedirs("./Partie 1/modele2", exist_ok=True)

# Chargement des données NumPy déjà normalisées [0, 1] et aplaties (784)
X_train = np.load("./Partie 1/saves/X_train.npy")
y_train = np.load("./Partie 1/saves/y_train.npy")
X_test  = np.load("./Partie 1/saves/X_test.npy")
y_test  = np.load("./Partie 1/saves/y_test.npy")

n_train, d = X_train.shape   # (60000, 784)
n_test     = X_test.shape[0] # (10000)[cite: 1]
K          = 10              # Chiffres 0 à 9

print(f"Données chargées avec succès !")
print(f"Train : {n_train} exemples | Test : {n_test} exemples | Dimension : {d}")


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    """Transforme les étiquettes (0-9) en vecteurs de probabilité binaire[cite: 2]."""
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y

Y_train = one_hot(y_train, K)
Y_test  = one_hot(y_test,  K)


# ─────────────────────────────────────────────
# 2. Fonctions d'activation
# ─────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    """φ(x) = max(0, x)"""
    return np.maximum(0.0, x)

def relu_deriv(x: np.ndarray) -> np.ndarray:
    """φ'(x) = 1 si x > 0, 0 sinon"""
    return (x > 0).astype(float)

def softmax(O: np.ndarray) -> np.ndarray:
    """Softmax numériquement stable, appliqué ligne par ligne."""
    O_s = O - O.max(axis=1, keepdims=True)
    E = np.exp(O_s)
    return E / E.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
# 3. Classe MLP générique (H = 1 ou H = 2 couches cachées)
# ─────────────────────────────────────────────

class MLP:
    """
    Réseau de neurones dense multi-couches.

    Architecture (exemple H=2) :
      Entrée (784)
        → Couche 1 cachée : o1 = z0 A1^T + b1,  z1 = ReLU(o1)   [p1 neurones]
        → Couche 2 cachée : o2 = z1 A2^T + b2,  z2 = ReLU(o2)   [p2 neurones]
        → Couche sortie   : o  = z2 A3^T + b3,  P  = Softmax(o)  [10 neurones]

    Paramètres :
      layer_sizes : liste des tailles de couches cachées, ex. [128] ou [256, 128]
      lr          : learning rate
    """

    def __init__(self, input_dim: int, layer_sizes: list, output_dim: int,
                 lr: float = 0.01, seed: int = 42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.H = len(layer_sizes)          # nombre de couches cachées

        # Initialisation He (adaptée à ReLU) : σ = sqrt(2 / fan_in)
        rng = np.random.default_rng(seed)
        dims = [input_dim] + layer_sizes + [output_dim]
        self.W = []   # poids : W[h] ∈ R^{dims[h+1] x dims[h]}
        self.b = []   # biais  : b[h] ∈ R^{dims[h+1]}
        for h in range(len(dims) - 1):
            fan_in = dims[h]
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, std, (dims[h+1], dims[h])))
            self.b.append(np.zeros(dims[h+1]))

    # ── Passe avant ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray):
        """
        Calcule la sortie du réseau et mémorise les activations
        pour la rétropropagation.

        Stocke :
          self.Z  : activations après φ  (incluant l'entrée en Z[0])
          self.O  : logits avant φ (pour chaque couche cachée)
        """
        self.Z = [X]            # Z[0] = entrée, Z[h] = sortie couche h
        self.O_hidden = []      # logits des couches cachées (avant ReLU)

        Z = X
        # Couches cachées : activation ReLU
        for h in range(self.H):
            O_h = Z @ self.W[h].T + self.b[h]   # (n, p_h)
            self.O_hidden.append(O_h)
            Z = relu(O_h)                         # z^h = φ(o^h)
            self.Z.append(Z)

        # Couche de sortie : logits bruts (softmax appliqué dans la loss)
        O_out = Z @ self.W[-1].T + self.b[-1]    # (n, K)
        P = softmax(O_out)
        self.P = P
        return P

    # ── Rétropropagation ─────────────────────────────────────────────────────

    def backward(self, Y: np.ndarray):
        """
        Calcule les gradients par rétropropagation (chain rule).

        Gradient de la cross-entropy ∘ softmax sur la dernière couche :
          δ_out = (P - Y) / n

        Pour chaque couche cachée h (de la dernière à la première) :
          δ_h = (δ_{h+1} W_{h+1}) ⊙ φ'(O_h)

        Gradients des paramètres :
          dW[h] = δ_{h+1}^T Z[h]
          db[h] = sum(δ_{h+1}, axis=0)
        """
        n = Y.shape[0]
        dW = [None] * (self.H + 1)
        db = [None] * (self.H + 1)

        # Gradient sur la couche de sortie
        delta = (self.P - Y) / n             # (n, K)

        dW[-1] = delta.T @ self.Z[-1]        # (K, p_H)
        db[-1] = delta.sum(axis=0)           # (K,)

        # Rétropropagation à travers les couches cachées
        for h in range(self.H - 1, -1, -1):
            # Propagation du gradient à travers W de la couche suivante
            delta = delta @ self.W[h+1]      # (n, p_h)
            # Multiplication par la dérivée de ReLU (chain rule)
            delta = delta * relu_deriv(self.O_hidden[h])   # (n, p_h)

            dW[h] = delta.T @ self.Z[h]      # (p_h, dims[h])
            db[h] = delta.sum(axis=0)        # (p_h,)

        return dW, db

    # ── Mise à jour SGD ───────────────────────────────────────────────────────

    def step(self, dW: list, db: list):
        """Mise à jour des paramètres : descente de gradient."""
        for h in range(len(self.W)):
            self.W[h] -= self.lr * dW[h]
            self.b[h]  -= self.lr * db[h]

    # ── Prédiction et métriques ───────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.forward(X)
        return np.argmax(P, axis=1)

    def error_rate(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) != y)

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        P = self.forward(X)
        eps = 1e-12
        return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


# ─────────────────────────────────────────────
# 4. Boucle d'entraînement
# ─────────────────────────────────────────────

def train(model: MLP,
          X_train, Y_train, y_train,
          X_test,  y_test,
          n_epochs: int = 30,
          batch_size: int = 256):
    """
    Entraînement SGD mini-batch.
    Retourne l'historique des métriques.
    """
    n = X_train.shape[0]
    history = {"loss_train": [], "err_train": [], "err_test": []}

    for epoch in range(n_epochs):
        # Mélange aléatoire
        idx = np.random.permutation(n)
        X_s, Y_s = X_train[idx], Y_train[idx]

        # Mini-batches
        for start in range(0, n, batch_size):
            Xb = X_s[start:start + batch_size]
            Yb = Y_s[start:start + batch_size]
            model.forward(Xb)
            dW, db = model.backward(Yb)
            model.step(dW, db)

        # Métriques (sur tout le train pour la courbe)
        loss    = model.loss(X_train, Y_train)
        err_tr  = model.error_rate(X_train, y_train)
        err_te  = model.error_rate(X_test,  y_test)

        history["loss_train"].append(loss)
        history["err_train"].append(err_tr)
        history["err_test"].append(err_te)

        print(f"  Époque {epoch+1:3d}/{n_epochs} | "
              f"Loss : {loss:.4f} | "
              f"Err train : {100*err_tr:.2f}% | "
              f"Err test : {100*err_te:.2f}%")

    return history


# ─────────────────────────────────────────────
# 5. Entraînement des deux architectures
# ─────────────────────────────────────────────

results = {}

# ── H = 1 couche cachée (p1 = 256 neurones) ──────────────────────────────────
print("\n=== MLP — 1 couche cachée (p1=256, ReLU) ===")
mlp1 = MLP(input_dim=d, layer_sizes=[256], output_dim=K, lr=0.1)
h1 = train(mlp1, X_train, Y_train, y_train, X_test, y_test,
           n_epochs=30, batch_size=256)
results["H=1 (256)"] = {
    "history":   h1,
    "err_train": mlp1.error_rate(X_train, y_train),
    "err_test":  mlp1.error_rate(X_test,  y_test),
}

# ── H = 2 couches cachées (p1=256, p2=128 neurones) ──────────────────────────
print("\n=== MLP — 2 couches cachées (p1=256, p2=128, ReLU) ===")
mlp2 = MLP(input_dim=d, layer_sizes=[256, 128], output_dim=K, lr=0.1)
h2 = train(mlp2, X_train, Y_train, y_train, X_test, y_test,
           n_epochs=30, batch_size=256)
results["H=2 (256-128)"] = {
    "history":   h2,
    "err_train": mlp2.error_rate(X_train, y_train),
    "err_test":  mlp2.error_rate(X_test,  y_test),
}


# ─────────────────────────────────────────────
# 6. Résumé comparatif
# ─────────────────────────────────────────────

print("\n" + "="*55)
print(f"{'Modèle':<25} {'Err train':>10} {'Err test':>10}")
print("="*55)
# Résultats du modèle linéaire (partie 1.2.1) — à renseigner
print(f"{'Linéaire (1.2.1)':<25} {'~8%':>10} {'~8%':>10}")
for name, r in results.items():
    print(f"{name:<25} {100*r['err_train']:>9.2f}% {100*r['err_test']:>9.2f}%")
print("="*55)


# ─────────────────────────────────────────────
# 7. Courbes d'apprentissage comparatives
# ─────────────────────────────────────────────

colors = {"H=1 (256)": "steelblue", "H=2 (256-128)": "tomato"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, r in results.items():
    epochs = range(1, len(r["history"]["loss_train"]) + 1)
    axes[0].plot(epochs, r["history"]["loss_train"],
                 label=name, color=colors[name])
    axes[1].plot(epochs, [100*e for e in r["history"]["err_train"]],
                 label=f"{name} — train", color=colors[name], linestyle="--")
    axes[1].plot(epochs, [100*e for e in r["history"]["err_test"]],
                 label=f"{name} — test",  color=colors[name])

axes[0].set_title("Cross-entropy (train)")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Taux d'erreur (%)")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Erreur (%)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Comparaison MLP H=1 vs H=2 sur MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("./Partie 1/modele2/mnist_mlp_curves.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 8. Analyse des erreurs (chiffres mal classés)
# ─────────────────────────────────────────────

print("\n=== Analyse des erreurs (MLP H=2) ===")
y_pred = mlp2.predict(X_test)
wrong_idx = np.where(y_pred != y_test)[0]
print(f"Nombre d'erreurs : {len(wrong_idx)} / {len(y_test)}")

# Matrice de confusion
conf = np.zeros((K, K), dtype=int)
for true, pred in zip(y_test, y_pred):
    conf[true][pred] += 1

print("\nMatrice de confusion :")
header = "     " + "  ".join(f"{k:3d}" for k in range(K))
print(header)
for k in range(K):
    row = f"  {k}  " + "  ".join(f"{conf[k,j]:3d}" for j in range(K))
    print(row)

# Visualisation de quelques erreurs
fig, axes = plt.subplots(3, 6, figsize=(13, 7))
axes = axes.flat
rng2 = np.random.default_rng(7)
sample = rng2.choice(wrong_idx, size=18, replace=False)

for ax, idx in zip(axes, sample):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Vrai:{y_test[idx]}  Prédit:{y_pred[idx]}", fontsize=8)
    ax.axis("off")

plt.suptitle("Exemples mal classés — MLP H=2", fontsize=13)
plt.tight_layout()
plt.savefig("./Partie 1/modele2/mnist_mlp_errors.png", dpi=150)
plt.show()
print("Figures sauvegardées.")


# ─────────────────────────────────────────────
# 9. Réduction en 2D pour visualisation (PCA manuelle)
# ─────────────────────────────────────────────
# On centre les données et on projette sur les 2 premières
# composantes principales pour illustrer la séparabilité.

print("\n=== Projection PCA 2D ===")
n_vis = 3000
idx_vis = np.random.choice(len(X_test), n_vis, replace=False)
X_vis = X_test[idx_vis]
y_vis = y_test[idx_vis]

# PCA manuelle
X_c = X_vis - X_vis.mean(axis=0)
cov = (X_c.T @ X_c) / n_vis
vals, vecs = np.linalg.eigh(cov)
# Trier par valeur propre décroissante
order = np.argsort(vals)[::-1]
PC = vecs[:, order[:2]]          # (784, 2)
X_2d = X_c @ PC                  # (n_vis, 2)

fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for k in range(K):
    mask = y_vis == k
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               s=8, alpha=0.6, color=cmap(k), label=str(k))
ax.set_title("Projection PCA 2D de MNIST (test, 3000 points)")
ax.legend(title="Chiffre", markerscale=3, fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("./Partie 1/modele2/mnist_pca2d.png", dpi=150)
plt.show()
print("PCA 2D sauvegardée.")