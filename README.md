# MolGAN guidé par Tox21

Projet pédagogique de génération de petits graphes moléculaires avec le
`BasicMolGANModel` PyTorch de DeepChem. Un ensemble de classifieurs entraînés sur
les 12 essais Tox21 estime la probabilité de toxicité, utilisée comme récompense
de policy gradient pour guider le générateur.

## Structure

```text
configs/       Paramètres par défaut
data/raw/      CSV source Tox21 (non versionné)
data/processed/ CSV nettoyés et filtrés (non versionnés)
notebooks/     Exploration et démonstrations
src/data/      Chargement, nettoyage et préparation
src/models/    MolGAN et prédicteurs de toxicité
src/rewards/   Récompense et fine-tuning
src/evaluation/ Métriques de génération
src/visualization/ Images RDKit
tests/         Tests unitaires
scripts/       Préparation, entraînement et évaluation
artifacts/     Modèles et rapports produits (non versionnés)
```

## Installation

Python 3.10 ou plus récent est requis. PyTorch CPU est suffisant pour tester;
un entraînement long est plus adapté à un environnement GPU ou à Google Colab.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Placez le fichier Tox21 à `data/raw/tox21.csv`. La source est conservée telle
quelle et les fichiers dans `data/` comme dans `artifacts/` ne sont pas versionnés.

## 1. Nettoyer et analyser les données

```powershell
python scripts/prepare_data.py
```

Le script canonise les SMILES valides, enlève les structures invalides, dédoublonne
par structure canonique et conserve les valeurs d'essai manquantes. En cas de
labels contradictoires pour une même structure/cible, le label de cette cible
devient manquant. Il écrit :

- `data/processed/tox21_clean.csv`
- `data/processed/tox21_molgan_compatible.csv`
- `data/processed/tox21_analysis.json`

Le MolGAN DeepChem configuré ici accepte au maximum 9 atomes et les éléments
C/N/O/F. Le sous-ensemble compatible est donc plus petit que Tox21 complet.

## 2. Entraîner, guider et générer

```powershell
python scripts/train.py
```

Les hyperparamètres se trouvent dans `configs/default.json`. Exemple de test
rapide (à ne pas interpréter comme un entraînement convergé) :

```powershell
python scripts/train.py --limit 64 --epochs 1 --candidate-pool 32 `
  --n-samples 5 --reward-steps 2
```

Le flux d'entraînement entraîne d'abord un modèle Random Forest par essai Tox21
sur des empreintes Morgan, puis le GAN sur les structures compatibles. Une étape
de policy gradient maximise `1 - poids_toxicité × risque_prédit` (les graphes
invalides reçoivent une récompense négative). Les molécules générées sont
évaluées par validité, unicité, diversité (similarité de Tanimoto) et risque
prédit; les échantillons les moins risqués sont enregistrés.

Pour accentuer ou diminuer la pénalité prédite :

```powershell
python scripts/train.py --toxicity-weight 2.0
```

`toxicity-weight=0` désactive cette pénalité. Il s'agit d'une optimisation selon
des prédictions in silico, pas d'une preuve de sécurité ou d'une évaluation
toxicologique expérimentale. Les résultats dépendent de l'entraînement du GAN
et de la qualité des prédicteurs.

Sorties principales dans `artifacts/` : générateur PyTorch, classifieurs Tox21,
molécules au format CSV, image RDKit et métriques JSON.

## 3. Réévaluer des molécules sauvegardées

```powershell
python scripts/evaluate.py
```

## Tests

```powershell
python -m pytest
```
