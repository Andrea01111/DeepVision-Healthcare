import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import os

os.makedirs("./Partie 2/saves",   exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. Chargement des données CIFAR-10
# ═══════════════════════════════════════════════════════════════

data_dir = "./data_cifar10"
train_raw = datasets.CIFAR10(root=data_dir, train=True,  download=True)
test_raw  = datasets.CIFAR10(root=data_dir, train=False, download=True)

# CIFAR-10 → numpy : shape (N, 32, 32, 3), valeurs uint8 [0, 255]
X_train_rgb = np.array(train_raw.data, dtype=np.float64) / 255.0  # (50000, 32, 32, 3)
X_test_rgb  = np.array(test_raw.data,  dtype=np.float64) / 255.0  # (10000, 32, 32, 3)
y_train = np.array(train_raw.targets)
y_test  = np.array(test_raw.targets)




# ═══════════════════════════════════════════════════════════════
# 2. Sauvegarde (optionnelle)
# ═══════════════════════════════════════════════════════════════

os.makedirs("./Partie 2/saves", exist_ok=True)
np.save("./Partie 2/saves/X_train_rgb.npy", X_train_rgb)
np.save("./Partie 2/saves/y_train.npy", y_train)
np.save("./Partie 2/saves/X_test_rgb.npy",  X_test_rgb)
np.save("./Partie 2/saves/y_test.npy",  y_test)
print("\nDonnées sauvegardées dans ./Partie 2/saves/")