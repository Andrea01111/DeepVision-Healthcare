"""
Projet SM604 - Partie 1.2.1 : Modèle linéaire multi-classe sur MNIST
Version compatible avec les données sauvegardées (NumPy)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 1. Chargement des données (Depuis le dossier ./saves)
# ─────────────────────────────────────────────
print("Chargement des données MNIST depuis ./Partie 1/saves...")

# On vérifie si les fichiers existent
if not os.path.exists("./Partie 1/saves/X_train.npy"):
    raise FileNotFoundError("Erreur : Lance d'abord le script d'Étape 1.1 pour générer les fichiers .npy")
os.makedirs("./Partie 1/modele1", exist_ok=True)

X_train = np.load("./Partie 1/saves/X_train.npy")
y_train = np.load("./Partie 1/saves/y_train.npy")
X_test  = np.load("./Partie 1/saves/X_test.npy")
y_test  = np.load("./Partie 1/saves/y_test.npy")

n_train, d = X_train.shape   # (60000, 784)
n_test     = X_test.shape[0]
K          = 10               # nombre de classes {0,...,9}

print(f"Données chargées !")
print(f"Train : {n_train} exemples | Test : {n_test} exemples | Dimension : {d}")

# ─────────────────────────────────────────────
# 2. Encodage one-hot des étiquettes
# ─────────────────────────────────────────────
def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y

Y_train = one_hot(y_train, K)
Y_test  = one_hot(y_test,  K)

# ─────────────────────────────────────────────
# 3. Initialisation des paramètres
# ─────────────────────────────────────────────
# Modèle : o = A x + b
#   A ∈ R^{10 x 784},  b ∈ R^{10}
rng = np.random.default_rng(0)
A = rng.normal(0, 0.01, (K, d))   # Matrice de poids (10, 784)
b = np.zeros(K)                  # Biais (10,)

# ─────────────────────────────────────────────
# 4. Fonctions utilitaires (Softmax, Cross-Entropy, Forward)
# ─────────────────────────────────────────────
def softmax(O):
    O_shifted = O - O.max(axis=1, keepdims=True)
    E = np.exp(O_shifted)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Y, P):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

def forward(X, A, b):
    O = X @ A.T + b
    P = softmax(O)
    return O, P

def predict(X, A, b):
    _, P = forward(X, A, b)
    return np.argmax(P, axis=1)

def error_rate(X, y, A, b):
    y_hat = predict(X, A, b)
    return np.mean(y_hat != y)

# ─────────────────────────────────────────────
# 5. Calcul des gradients
# ─────────────────────────────────────────────

# Démonstration du gradient (à connaître pour la soutenance) :
#
#   dL/dO = (P - Y) / n         ∈ R^{n x K}
#
#   dL/dA = (P - Y)^T X / n    ∈ R^{K x d}
#   dL/db = mean(P - Y, axis=0) ∈ R^K
#
# La dérivée vient de la composition softmax ∘ cross-entropy.

def gradients(X, Y, A, b):
    n = X.shape[0]
    _, P = forward(X, A, b)
    delta = (P - Y) / n
    dA = delta.T @ X
    db = delta.sum(axis=0)
    return dA, db

# ─────────────────────────────────────────────
# 6. Descente de gradient (mini-batch SGD)
# ─────────────────────────────────────────────
def sgd(X_train, Y_train, y_train, X_test, y_test, A_init, b_init, lr=0.1, n_epochs=30, batch_size=256):
    A, b = A_init.copy(), b_init.copy()
    n = X_train.shape[0]
    history = {"loss_train": [], "err_train": [], "err_test": []}

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        X_s, Y_s = X_train[idx], Y_train[idx]

        for start in range(0, n, batch_size):
            Xb = X_s[start:start + batch_size]
            Yb = Y_s[start:start + batch_size]
            dA, db = gradients(Xb, Yb, A, b)
            A -= lr * dA
            b -= lr * db

        _, P_train = forward(X_train, A, b)
        loss = cross_entropy(Y_train, P_train)
        err_tr = error_rate(X_train, y_train, A, b)
        err_te = error_rate(X_test,  y_test,  A, b)

        history["loss_train"].append(loss)
        history["err_train"].append(err_tr)
        history["err_test"].append(err_te)

        print(f"Époque {epoch+1:2d}/{n_epochs} | Loss: {loss:.4f} | Err train: {100*err_tr:.2f}% | Err test: {100*err_te:.2f}%")

    return A, b, history

# ─────────────────────────────────────────────
# 7. Lancement de l'entraînement
# ─────────────────────────────────────────────
print("\n=== Début de l'entraînement ===")
A_trained, b_trained, history = sgd(X_train, Y_train, y_train, X_test, y_test, A, b)

# ─────────────────────────────────────────────
# 8. Visualisation et Sauvegarde des courbes
# ─────────────────────────────────────────────
epochs = range(1, len(history["loss_train"]) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["loss_train"])
plt.title("Évolution de la Perte (Loss)")
plt.xlabel("Époque")

plt.subplot(1, 2, 2)
plt.plot(epochs, [100*e for e in history["err_train"]], label="Train")
plt.plot(epochs, [100*e for e in history["err_test"]], label="Test")
plt.title("Évolution du Taux d'Erreur (%)")
plt.legend()

plt.tight_layout()
plt.savefig("./Partie 1/modele1/courbes_apprentissage.png")
plt.show()

# ─────────────────────────────────────────────
# 9. Visualisation des poids (A)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for k, ax in enumerate(axes.flat):
    img = A_trained[k].reshape(28, 28)
    ax.imshow(img, cmap="RdBu_r")
    ax.set_title(f"Poids Chiffre {k}")
    ax.axis("off")
plt.suptitle("Visualisation des poids appris par le modèle linéaire")
plt.savefig("./Partie 1/modele1/poids_mnist.png")
plt.show()