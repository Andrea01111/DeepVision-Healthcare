# 🧠 DeepVision-Healthcare : De MNIST aux CNN (Classification d'Images) 🚀 *(Work in Progress)*

## 📌 Contexte du Projet
Ce dépôt contient mon projet réalisé dans le cadre du module **Mathématiques pour le Machine Learning (SM604 - EFREI)**. 

L'objectif de ce projet est d'explorer progressivement les méthodes d'apprentissage supervisé pour la classification d'images. Le parcours commence par des implémentations mathématiques *from scratch* (NumPy) sur des jeux de données simples, pour évoluer vers l'utilisation de réseaux de neurones convolutifs (CNN) via PyTorch sur des images complexes.

> ⚠️ **Note :** Ce projet est actuellement **en cours de développement (Ongoing)**. Les parties 1 (MNIST) et 2 (CIFAR-10) sont terminées. La partie 3, qui constitue l'aboutissement du projet (détection sur images médicales), est en cours de réalisation.

---

## 🏗️ Architecture du Dépôt

Le projet est divisé en plusieurs parties évolutives :

### 📂 Partie 1 : Classification de chiffres manuscrits (MNIST) - *Terminé*
Implémentation **100% from scratch** (sans framework de Deep Learning, uniquement avec NumPy) pour comprendre les mathématiques sous-jacentes (Forward pass, Backpropagation, Cross-Entropy).
* **`mnist_model.py`** : Chargement, normalisation et aplatissement des données MNIST.
* **`modele1.py`** : Modèle linéaire multi-classes entraîné par descente de gradient stochastique (SGD).
* **`modele2.py`** : Réseau de neurones multi-couches (MLP) avec 1 ou 2 couches cachées, activation ReLU et Softmax. Inclut une visualisation par réduction de dimensionnalité (PCA).

### 📂 Partie 2 : Classification d'images en couleur (CIFAR-10) - *Terminé*
Montée en complexité avec des images RGB (3 canaux) et introduction aux frameworks modernes.
* **`cifar10_model.py`** : Chargement et préparation du dataset CIFAR-10.
* **`preliminaire.py`** : Test des limites des modèles précédents (Linéaire et MLP "from scratch") sur des images complexes.
* **`cnn.py`** : Implémentation d'un **Réseau de Neurones Convolutifs (CNN)** complet avec **PyTorch**.
* **`cnn_cifar10.pth`** : Poids sauvegardés du modèle CNN entraîné.

### 📂 Partie 3 : Détection de cancers du sein (Mammographies) - *À venir*
*(Dossier en cours de construction)* : Application des CNN sur un jeu de données médicales réel pour détecter des signes de cancers.

---

## 🛠️ Stack Technique & Bibliothèques

* **Langage :** Python 3
* **Machine Learning "From Scratch" :** NumPy, Mathématiques (Calcul matriciel, Dérivation en chaîne)
* **Deep Learning :** PyTorch (`torch`, `torch.nn`, `torchvision`)
* **Analyse & Dataviz :** Matplotlib, Seaborn, Scikit-learn

---

## ⏳ To-Do List & Prochaines Étapes

- [x] **Étape 1 :** Implémentation *from scratch* du modèle Linéaire et MLP sur MNIST.
- [x] **Étape 2 :** Transition vers PyTorch et implémentation d'un CNN sur CIFAR-10.
- [ ] **Étape 3 :** Chargement et prétraitement du dataset médical (Mammographies).
- [ ] **Étape 3 :** Adaptation de l'architecture CNN pour la classification binaire (Sain / Malade).
- [ ] **Étape 3 :** Évaluation des performances cliniques (Rappel/Sensibilité pour minimiser les faux négatifs).

---

## 🚀 Installation & Exécution

### 1. Prérequis
Assurez-vous d'avoir installé les bibliothèques nécessaires :
```bash
pip install numpy matplotlib scikit-learn torch torchvision seaborn
