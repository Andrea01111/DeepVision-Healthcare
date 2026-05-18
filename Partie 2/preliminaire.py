"""
Projet SM604 - Partie 2.2 : Travail préliminaire sur CIFAR-10
-------------------------------------------------------------
On réutilise les architectures de la Partie 1 (modèle linéaire + MLP), en les adaptant à CIFAR-10.

Deux études :
  A) Images converties en niveaux de gris  → vecteur x ∈ R^1024
  B) Images en couleur (RGB aplaties)      → vecteur x ∈ R^3072

Pour chaque étude, on évalue :
  - le modèle linéaire (Partie 1.2.1)
  - le MLP H=1 (Partie 1.2.2)
  - le MLP H=2 (Partie 1.2.2)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("./Partie 2/resultats", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Chargement des données CIFAR-10
# ═══════════════════════════════════════════════════════════════

print("Chargement des données CIFAR-10 depuis ./Partie 2/saves...")

# On vérifie si les fichiers existent
if not os.path.exists("./Partie 2/saves/X_train_rgb.npy"):
    raise FileNotFoundError("Erreur : le fichier n'existe pas")
os.makedirs("./Partie 2/preliminaire", exist_ok=True)

X_train_rgb = np.load("./Partie 2/saves/X_train_rgb.npy")
y_train = np.load("./Partie 2/saves/y_train.npy")
X_test_rgb = np.load("./Partie 2/saves/X_test_rgb.npy")
y_test  = np.load("./Partie 2/saves/y_test.npy")


CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
K = 10


print(f"Données chargées !")

print(f"CIFAR-10 chargé : train={X_train_rgb.shape}, test={X_test_rgb.shape}")


# ═══════════════════════════════════════════════════════════════
# 2. Préparation des données — Étude A : niveaux de gris
# ═══════════════════════════════════════════════════════════════
# Formule standard : x_j = 0.299 R_j + 0.587 G_j + 0.114 B_j
# On obtient x ∈ R^1024  (32×32)

def to_grayscale(X_rgb):
    """X_rgb : (N, 32, 32, 3) → X_gray : (N, 1024)"""
    coefs = np.array([0.299, 0.587, 0.114])
    gray = X_rgb @ coefs           # (N, 32, 32)
    return gray.reshape(len(X_rgb), -1)  # (N, 1024)

X_train_gray = to_grayscale(X_train_rgb)   # (50000, 1024)
X_test_gray  = to_grayscale(X_test_rgb)    # (10000, 1024)

# ═══════════════════════════════════════════════════════════════
# 3. Préparation des données — Étude B : couleur (RGB)
# ═══════════════════════════════════════════════════════════════
# x ∈ R^3072  (32×32×3)

X_train_color = X_train_rgb.reshape(len(X_train_rgb), -1)   # (50000, 3072)
X_test_color  = X_test_rgb.reshape(len(X_test_rgb),  -1)    # (10000, 3072)

# ═══════════════════════════════════════════════════════════════
# 4. Encodage one-hot
# ═══════════════════════════════════════════════════════════════

def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y

Y_train = one_hot(y_train, K)
Y_test  = one_hot(y_test,  K)

# ═══════════════════════════════════════════════════════════════
# 5. Fonctions communes (softmax, cross-entropy)
# ═══════════════════════════════════════════════════════════════

def softmax(O):
    O_s = O - O.max(axis=1, keepdims=True)
    E   = np.exp(O_s)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Y, P):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# ═══════════════════════════════════════════════════════════════
# 6. Modèle linéaire (Partie 1.2.1 — adapté)
# ═══════════════════════════════════════════════════════════════

def train_linear(X_train, Y_train, y_train, X_test, y_test,
                 lr=0.1, n_epochs=30, batch_size=256, seed=0):
    d = X_train.shape[1]
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.01, (K, d))
    b = np.zeros(K)

    history = {"loss": [], "err_train": [], "err_test": []}
    n = len(X_train)

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        Xs, Ys = X_train[idx], Y_train[idx]

        for start in range(0, n, batch_size):
            Xb = Xs[start:start + batch_size]
            Yb = Ys[start:start + batch_size]
            nb = Xb.shape[0]
            O  = Xb @ A.T + b
            P  = softmax(O)
            delta = (P - Yb) / nb
            A -= lr * (delta.T @ Xb)
            b -= lr * delta.sum(axis=0)

        # Métriques
        P_tr = softmax(X_train @ A.T + b)
        loss = cross_entropy(Y_train, P_tr)
        err_tr = np.mean(np.argmax(P_tr, axis=1) != y_train)
        P_te   = softmax(X_test  @ A.T + b)
        err_te = np.mean(np.argmax(P_te, axis=1) != y_test)

        history["loss"].append(loss)
        history["err_train"].append(err_tr)
        history["err_test"].append(err_te)
        print(f"  [Linéaire] Ép.{epoch+1:2d}/{n_epochs} | "
              f"Loss:{loss:.4f} | Train:{100*err_tr:.1f}% | Test:{100*err_te:.1f}%")

    return A, b, history

# ═══════════════════════════════════════════════════════════════
# 7. MLP générique (Partie 1.2.2 — identique, réutilisé tel quel)
# ═══════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, input_dim, layer_sizes, output_dim, lr=0.01, seed=42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.H  = len(layer_sizes)
        rng  = np.random.default_rng(seed)
        dims = [input_dim] + layer_sizes + [output_dim]
        self.W = []
        self.b = []
        for h in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[h])
            self.W.append(rng.normal(0, std, (dims[h+1], dims[h])))
            self.b.append(np.zeros(dims[h+1]))

    def forward(self, X):
        self.Z        = [X]
        self.O_hidden = []
        Z = X
        for h in range(self.H):
            O_h = Z @ self.W[h].T + self.b[h]
            self.O_hidden.append(O_h)
            Z = relu(O_h)
            self.Z.append(Z)
        O_out = Z @ self.W[-1].T + self.b[-1]
        self.P = softmax(O_out)
        return self.P

    def backward(self, Y):
        n  = Y.shape[0]
        dW = [None] * (self.H + 1)
        db = [None] * (self.H + 1)
        delta     = (self.P - Y) / n
        dW[-1]    = delta.T @ self.Z[-1]
        db[-1]    = delta.sum(axis=0)
        for h in range(self.H - 1, -1, -1):
            delta  = delta @ self.W[h+1]
            delta  = delta * relu_deriv(self.O_hidden[h])
            dW[h]  = delta.T @ self.Z[h]
            db[h]  = delta.sum(axis=0)
        return dW, db

    def step(self, dW, db):
        for h in range(len(self.W)):
            self.W[h] -= self.lr * dW[h]
            self.b[h]  -= self.lr * db[h]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def error_rate(self, X, y):
        return np.mean(self.predict(X) != y)

    def loss(self, X, Y):
        P   = self.forward(X)
        eps = 1e-12
        return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


def train_mlp(model, X_train, Y_train, y_train, X_test, y_test,
              n_epochs=30, batch_size=256, label="MLP"):
    n = len(X_train)
    history = {"loss": [], "err_train": [], "err_test": []}

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        Xs, Ys = X_train[idx], Y_train[idx]
        for start in range(0, n, batch_size):
            Xb = Xs[start:start + batch_size]
            Yb = Ys[start:start + batch_size]
            model.forward(Xb)
            dW, db = model.backward(Yb)
            model.step(dW, db)

        loss   = model.loss(X_train, Y_train)
        err_tr = model.error_rate(X_train, y_train)
        err_te = model.error_rate(X_test,  y_test)
        history["loss"].append(loss)
        history["err_train"].append(err_tr)
        history["err_test"].append(err_te)
        print(f"  [{label}] Ép.{epoch+1:2d}/{n_epochs} | "
              f"Loss:{loss:.4f} | Train:{100*err_tr:.1f}% | Test:{100*err_te:.1f}%")

    return history

# ═══════════════════════════════════════════════════════════════
# 8. Entraînement — Étude A : niveaux de gris (d=1024)
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("ÉTUDE A — Images en niveaux de gris (d=1024)")
print("═"*60)

d_gray = X_train_gray.shape[1]   # 1024

# 8a. Modèle linéaire
print("\n--- Modèle linéaire ---")
_, _, h_lin_gray = train_linear(X_train_gray, Y_train, y_train,
                                X_test_gray, y_test, lr=0.1, n_epochs=30)

# 8b. MLP H=1
print("\n--- MLP H=1 (256 neurones) ---")
mlp1_gray = MLP(input_dim=d_gray, layer_sizes=[256], output_dim=K, lr=0.05)
h_mlp1_gray = train_mlp(mlp1_gray, X_train_gray, Y_train, y_train,
                         X_test_gray, y_test, n_epochs=30, label="MLP H=1")

# 8c. MLP H=2
print("\n--- MLP H=2 (256-128 neurones) ---")
mlp2_gray = MLP(input_dim=d_gray, layer_sizes=[256, 128], output_dim=K, lr=0.05)
h_mlp2_gray = train_mlp(mlp2_gray, X_train_gray, Y_train, y_train,
                         X_test_gray, y_test, n_epochs=30, label="MLP H=2")

# ═══════════════════════════════════════════════════════════════
# 9. Entraînement — Étude B : couleur RGB (d=3072)
# ═══════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("ÉTUDE B — Images en couleur RGB (d=3072)")
print("═"*60)

d_color = X_train_color.shape[1]  # 3072

# 9a. Modèle linéaire
print("\n--- Modèle linéaire ---")
_, _, h_lin_color = train_linear(X_train_color, Y_train, y_train,
                                 X_test_color, y_test, lr=0.1, n_epochs=30)

# 9b. MLP H=1
print("\n--- MLP H=1 (256 neurones) ---")
mlp1_color = MLP(input_dim=d_color, layer_sizes=[256], output_dim=K, lr=0.05)
h_mlp1_color = train_mlp(mlp1_color, X_train_color, Y_train, y_train,
                          X_test_color, y_test, n_epochs=30, label="MLP H=1")

# 9c. MLP H=2
print("\n--- MLP H=2 (256-128 neurones) ---")
mlp2_color = MLP(input_dim=d_color, layer_sizes=[256, 128], output_dim=K, lr=0.05)
h_mlp2_color = train_mlp(mlp2_color, X_train_color, Y_train, y_train,
                          X_test_color, y_test, n_epochs=30, label="MLP H=2")

# ═══════════════════════════════════════════════════════════════
# 10. Résumé comparatif des taux d'erreur
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print(f"{'Modèle':<30} {'Entrée':<10} {'Err train':>10} {'Err test':>10}")
print("="*65)

# Résultats MNIST pour référence
print(f"{'Linéaire (MNIST)':<30} {'784':>10} {'~8%':>10} {'~8%':>10}")
print(f"{'MLP H=1 (MNIST)':<30} {'784':>10} {'~3%':>10} {'~3%':>10}")
print("-"*65)

results_summary = [
    ("Linéaire (gris)",   "1024",  h_lin_gray["err_train"][-1],  h_lin_gray["err_test"][-1]),
    ("MLP H=1 (gris)",    "1024",  h_mlp1_gray["err_train"][-1], h_mlp1_gray["err_test"][-1]),
    ("MLP H=2 (gris)",    "1024",  h_mlp2_gray["err_train"][-1], h_mlp2_gray["err_test"][-1]),
    ("Linéaire (couleur)","3072",  h_lin_color["err_train"][-1], h_lin_color["err_test"][-1]),
    ("MLP H=1 (couleur)", "3072",  h_mlp1_color["err_train"][-1],h_mlp1_color["err_test"][-1]),
    ("MLP H=2 (couleur)", "3072",  h_mlp2_color["err_train"][-1],h_mlp2_color["err_test"][-1]),
]

for name, dim, err_tr, err_te in results_summary:
    print(f"{name:<30} {dim:>10} {100*err_tr:>9.2f}% {100*err_te:>9.2f}%")

print("="*65)
print("\nRéférence littérature CIFAR-10 :")
print("  Convolutional Deep Belief Networks (2010) : 21.1%")
print("  Maxout Networks              (2013)       :  9.38%")
print("  ViT (An Image is Worth 16x16 Words, 2021) :  0.5%")

# ═══════════════════════════════════════════════════════════════
# 11. Visualisation — courbes d'apprentissage (test error)
# ═══════════════════════════════════════════════════════════════

epochs = range(1, 31)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Étude A : niveaux de gris
ax = axes[0]
ax.plot(epochs, [100*e for e in h_lin_gray["err_test"]],
        label="Linéaire", color="gray",      linestyle="--")
ax.plot(epochs, [100*e for e in h_mlp1_gray["err_test"]],
        label="MLP H=1",  color="steelblue")
ax.plot(epochs, [100*e for e in h_mlp2_gray["err_test"]],
        label="MLP H=2",  color="tomato")
ax.set_title("CIFAR-10 — Niveaux de gris (d=1024)\nTaux d'erreur test (%)")
ax.set_xlabel("Époque")
ax.set_ylabel("Erreur (%)")
ax.legend()
ax.grid(True, alpha=0.3)

# Étude B : couleur
ax = axes[1]
ax.plot(epochs, [100*e for e in h_lin_color["err_test"]],
        label="Linéaire", color="gray",      linestyle="--")
ax.plot(epochs, [100*e for e in h_mlp1_color["err_test"]],
        label="MLP H=1",  color="steelblue")
ax.plot(epochs, [100*e for e in h_mlp2_color["err_test"]],
        label="MLP H=2",  color="tomato")
ax.set_title("CIFAR-10 — Couleur RGB (d=3072)\nTaux d'erreur test (%)")
ax.set_xlabel("Époque")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Partie 2.2 — Travail préliminaire CIFAR-10", fontsize=13)
plt.tight_layout()
plt.savefig("./Partie 2/preliminaire/cifar10_courbes.png", dpi=150)
plt.show()
print("\nFigure sauvegardée : ./Partie 2/preliminaire/cifar10_courbes.png")

# ═══════════════════════════════════════════════════════════════
# 12. Visualisation de quelques exemples CIFAR-10
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Partie 2.2 — CIFAR-10 : un exemple par classe", fontsize=13)

for k, ax in enumerate(axes.flat):
    idx = np.where(y_train == k)[0][0]
    img = X_train_rgb[idx]              # shape (32, 32, 3)
    ax.imshow(img)
    ax.set_title(CLASSES[k], fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("./Partie 2/preliminaire/cifar10_exemples.png", dpi=100)
plt.show()
print("Figure sauvegardée : ./Partie 2/resultats/cifar10_exemples.png")