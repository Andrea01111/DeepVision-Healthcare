"""
Projet SM604 - Section 2.6 Option B
Architecture CNN sur CIFAR-10 avec PyTorch (rétropropagation automatique via Autograd)

Architecture (section 2.5) :
  Input (32,32,3)
  -> Conv1 : 64 filtres couleur       (32,32,64)
  -> Conv2 : 64 filtres 3D            (32,32,64)
  -> MaxPool 2x2                      (16,16,64)
  -> Conv3 : 64 filtres 3D            (16,16,64)
  -> MaxPool 2x2                      (8,8,64)
  -> Conv4 : 64 filtres 3D            (8,8,64)
  -> Flatten                          (4096,)
  -> Linear 4096 -> 10
  -> Softmax (via CrossEntropyLoss)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Dispositif utilisé : {DEVICE}")

BATCH_SIZE  = 64
EPOCHS      = 10
LEARNING_RATE = 0.001

CLASSES = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']

if __name__ == '__main__':
    # ─────────────────────────────────────────────
    # 2. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
    # ─────────────────────────────────────────────
    # Normalisation standard CIFAR-10 (moyenne et écart-type par canal RGB)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # légère augmentation
        transforms.RandomCrop(32, padding=4),       # légère augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std =(0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std =(0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True,
                                                download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False,
                                                download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Taille ensemble d'entraînement : {len(train_dataset)}")
    print(f"Taille ensemble de test        : {len(test_dataset)}")


    # ─────────────────────────────────────────────
    # 3. ARCHITECTURE CNN (section 2.5 du sujet)
    # ─────────────────────────────────────────────
    class CNN_CIFAR10(nn.Module):
        """
        Architecture convolutive conforme à la section 2.5 :
        Conv (64 filtres couleur)
        Conv3D (64 filtres 3D)
        MaxPool 2x2
        Conv3D (64 filtres 3D)
        MaxPool 2x2
        Conv3D (64 filtres 3D)
        Flatten -> 4096
        Dense  -> 10 classes
        """
        def __init__(self):
            super(CNN_CIFAR10, self).__init__()

            # Bloc convolutif 1 : deux convolutions successives
            # Conv1 : entrée 3 canaux (RGB) -> 64 feature maps  [section 2.5.2]
            # Conv2 : 64 -> 64 (filtres 3D)                     [section 2.5.3]
            self.bloc1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            # MaxPool 2x2 : (32,32,64) -> (16,16,64)            [section 2.5.4]
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Bloc convolutif 2 : une convolution 3D             [section 2.5.5]
            self.bloc2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            # MaxPool 2x2 : (16,16,64) -> (8,8,64)              [section 2.5.5]
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Bloc convolutif 3 : dernière convolution 3D        [section 2.5.5]
            self.bloc3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            # Aplatissement : 8*8*64 = 4096                      [section 2.4.3]
            # Couche dense : 4096 -> 10 classes                  [section 2.5.5]
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),              # régularisation contre l'overfitting
                nn.Linear(8 * 8 * 64, 10),     # 4096 -> 10
            )

        def forward(self, x):
            # Passage avant (forward pass)
            x = self.bloc1(x)        # (N, 3,  32, 32) -> (N, 64, 32, 32)
            x = self.pool1(x)        # (N, 64, 32, 32) -> (N, 64, 16, 16)
            x = self.bloc2(x)        # (N, 64, 16, 16) -> (N, 64, 16, 16)
            x = self.pool2(x)        # (N, 64, 16, 16) -> (N, 64,  8,  8)
            x = self.bloc3(x)        # (N, 64,  8,  8) -> (N, 64,  8,  8)
            x = torch.flatten(x, 1)  # (N, 64,  8,  8) -> (N, 4096)
            x = self.classifier(x)   # (N, 4096)        -> (N, 10)  [logits]
            return x                 # CrossEntropyLoss applique softmax en interne


    # ─────────────────────────────────────────────
    # 4. INITIALISATION DU MODÈLE
    # ─────────────────────────────────────────────
    model = CNN_CIFAR10().to(DEVICE)

    # Comptage des paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNombre de paramètres entraînables : {total_params:,}")
    print(model)

    # ─────────────────────────────────────────────
    # 5. FONCTION DE COÛT ET OPTIMISEUR
    # ─────────────────────────────────────────────
    # CrossEntropyLoss = LogSoftmax + NLLLoss  (équivalent à l'entropie croisée du cours)
    criterion = nn.CrossEntropyLoss()

    # Optimiseur Adam (plus stable que SGD pour démarrer)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler : réduit le LR si la loss stagne (aide contre l'overfitting)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=3)


    # ─────────────────────────────────────────────
    # 6. BOUCLE D'ENTRAÎNEMENT
    # ─────────────────────────────────────────────
    def train_one_epoch(model, loader, criterion, optimizer):
        """Effectue une époque d'entraînement et retourne la loss et l'accuracy."""
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()          # remise à zéro des gradients
            outputs = model(images)        # forward pass
            loss = criterion(outputs, labels)  # calcul de la loss
            loss.backward()                # rétropropagation (Autograd)
            optimizer.step()               # mise à jour des poids

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        return epoch_loss, epoch_acc


    def evaluate(model, loader, criterion):
        """Évalue le modèle sur un ensemble de données."""
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)

        return running_loss / total, correct / total


    # Historique pour les courbes
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print("\n─── Début de l'entraînement ───")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion)
        scheduler.step(test_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Époque {epoch:2d}/{EPOCHS} | "
            f"Loss entraîn.: {train_loss:.4f} | Acc entraîn.: {train_acc*100:.2f}% | "
            f"Loss test: {test_loss:.4f} | Acc test: {test_acc*100:.2f}%")

    print(f"\nTaux d'erreur final (entraînement) : {(1 - history['train_acc'][-1])*100:.2f}%")
    print(f"Taux d'erreur final (test)         : {(1 - history['test_acc'][-1])*100:.2f}%")


    # ─────────────────────────────────────────────
    # 7. COURBES D'APPRENTISSAGE
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, EPOCHS + 1)

    # Courbe de loss
    axes[0].plot(epochs_range, history['train_loss'], label='Entraînement', color='steelblue')
    axes[0].plot(epochs_range, history['test_loss'],  label='Test',         color='tomato')
    axes[0].set_title('Fonction de coût (Cross-Entropy)')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Courbe d'accuracy
    axes[1].plot(epochs_range, [a*100 for a in history['train_acc']], label='Entraînement', color='steelblue')
    axes[1].plot(epochs_range, [a*100 for a in history['test_acc']],  label='Test',         color='tomato')
    axes[1].set_title("Taux de bonne classification (%)")
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Courbes d\'apprentissage — CNN CIFAR-10 (Option B, PyTorch)', fontsize=13)
    plt.tight_layout()
    plt.savefig('./Partie 2/cnn/courbes_apprentissage_cifar10.png', dpi=150)
    plt.show()
    print("Courbes sauvegardées dans courbes_apprentissage_cifar10.png")


    # ─────────────────────────────────────────────
    # 8. MATRICE DE CONFUSION ET RAPPORT
    # ─────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('Matrice de confusion — CNN CIFAR-10')
    plt.tight_layout()
    plt.savefig('./Partie 2/cnn/matrice_confusion_cifar10.png', dpi=150)
    plt.show()
    print("Matrice de confusion sauvegardée dans ./Partie 2/cnn/matrice_confusion_cifar10.png")

    print("\n─── Rapport de classification ───")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))


    # ─────────────────────────────────────────────
    # 9. SAUVEGARDE DU MODÈLE
    # ─────────────────────────────────────────────
    torch.save(model.state_dict(), './Partie 2/cnn/cnn_cifar10.pth')
    print("Modèle sauvegardé dans ./Partie 2/cnn/cnn_cifar10.pth")

    # Pour recharger :
    # model = CNN_CIFAR10()
    # model.load_state_dict(torch.load('cnn_cifar10.pth'))
    # model.eval()

