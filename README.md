# MolGAN guidé par Tox21

Projet pédagogique de génération de petits graphes moléculaires depuis les
composés Tox21, avec le `BasicMolGANModel` PyTorch de DeepChem. Le baseline
adversarial est séparé de l'entraînement supervisé des prédicteurs et de toute
expérience de génération guidée par propriété.

## Structure

```text
configs/       Paramètres par défaut
data/raw/      CSV source Tox21 (non versionné)
data/deepchem_cache/ cache MoleculeNet et partitions (non versionné)
data/audit/    rapport de répartition et de labels (non versionné)
notebooks/     Exploration Tox21 et parcours complet du projet (Jupyter)
src/data/      Chargement Tox21, préparation et validation des graphes MolGAN
src/models/    MolGAN et prédicteurs d'activité Tox21
src/rewards/   Récompense et fine-tuning
src/evaluation/ Métriques de génération et évaluations Tox21 masquées
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

Pour exécuter le notebook de bout en bout avec un kernel dédié dans VS Code :

```powershell
python -m pip install -e ".[notebook]"
python -m ipykernel install --user --name molgan-tox21 --display-name "Python (MolGAN Tox21)"
```

Ouvrez `notebooks/02_project_end_to_end.ipynb` et sélectionnez le kernel
**Python (MolGAN Tox21)**. Les cellules d'entraînement long et de comparaison
sont désactivées par défaut.

Placez le fichier Tox21 à `data/raw/tox21.csv`. La source est conservée telle
quelle et les fichiers dans `data/` comme dans `artifacts/` ne sont pas versionnés.

## 1. Charger et auditer les partitions officielles Tox21

```powershell
python scripts/prepare_data.py
```

Le script utilise `deepchem.molnet.load_tox21` de DeepChem 2.8.0 avec
`featurizer="GraphConv"`, `splitter="scaffold"` et aucun transformer. Il conserve
les partitions train/validation/test données par MoleculeNet et met les données
téléchargées/featurisées en cache sous `data/deepchem_cache/`. La première
exécution récupère les données officielles si elles ne sont pas déjà en cache.

Un audit complet, avec le nombre de composés et les comptes positifs/négatifs/
masqués par cible et par partition, est écrit dans
`data/audit/tox21_molnet_audit.json`. Le loader retourne des objets `ConvMol`;
les identifiants `Dataset.ids` sont les SMILES des composés.

Les labels d'essai manquants sont masqués par `Dataset.w == 0`; `Dataset.y` peut
contenir zéro comme valeur de remplissage à ces positions. Les pertes et métriques
doivent utiliser `w` pour ne pas compter ces valeurs masquées comme des négatifs.
Le splitter scaffold de DeepChem 2.8.0 est déterministe et ignore sa graine
optionnelle; le pipeline conserve ce comportement officiel au lieu de créer un
split aléatoire différent. Le paramètre `seed=42` est utilisé pour les modèles
de toxicité, pas pour modifier les partitions.

## 2. Préparer et valider les graphes moléculaires

La représentation est celle de `deepchem.feat.MolGanFeaturizer` dans DeepChem
2.8.0, avec `max_atom_count=9` et `kekulize=True`, cohérente avec les paramètres
MolGAN par défaut :

| Tenseur | Forme discrète | Forme one-hot du discriminateur |
| --- | --- | --- |
| Atomes | `(B, 9)` | `(B, 9, 5)` |
| Liaisons/adjacence | `(B, 9, 9)` | `(B, 9, 9, 5)` |

Les indices d'atome correspondent aux labels `[0, 6, 7, 8, 9]` (padding, C, N,
O, F); les indices de liaison sont `[ZERO, SINGLE, DOUBLE, TRIPLE, AROMATIC]`.
La matrice d'adjacence est symétrique. Les entrées non nulles désignent un type
de liaison; `ZERO` signifie absence de liaison. DeepChem kekulize les entrées
aromatiques avant featurization; les anneaux sont reconstruits et resanitizés
par RDKit. Pendant la reconstruction, les arêtes incidentes aux sommets de
padding (catégorie atome `0`) sont retirées avant conversion et leur nombre est
inclus dans le rapport; les sommets actifs isolés et graphes disjoints sont
rejetés.

Le pipeline refuse explicitement les SMILES invalides, échecs de sanitation,
atomes non pris en charge, molécules de plus de 9 atomes (hydrogènes implicites
exclus), liaisons incompatibles,
charges formelles, isotopes, stéréochimie, fragments disjoints et atomes isolés.
Le modèle ne représente que C/N/O/F neutres, sans isotope ni stéréochimie, et
perdrait ces informations autrement. Une entrée est aussi rejetée si son SMILES
canonique diffère après reconstruction du graphe. Les formes correctes ne
suffisent pas à valider une molécule : les graphes générés sont convertis en
RDKit, sanitisés, puis convertis en SMILES canoniques; les taux et causes
d'échec sont rapportés.

DeepChem 2.8.0 expose le traitement inverse sous forme de `defeaturize`, qui
sanitise et nettoie ensemble. Pour séparer les compteurs de conversion et de
sanitation sans réimplémenter son décodage, le convertisseur retire le padding
et appelle l'implémentation inverse interne de la version épinglée avec
`sanitize=False, cleanup=False`, puis exécute `Chem.SanitizeMol` explicitement.
La version DeepChem est donc épinglée dans `pyproject.toml`.

Exécutez une démonstration déterministe sur les premiers SMILES du split
train officiel (aucun entraînement adversarial n'est lancé) :

```powershell
python scripts/demo_graph_preparation.py --count 12
```

Le script compare les SMILES canoniques d'origine et reconstruits, imprime le
taux de survie de conversion/sanitation et enregistre une grille RDKit dans
`artifacts/molgan_graph_demo.png`, ainsi que son rapport dans
`artifacts/molgan_graph_demo.json`. Les molécules incompatibles et leurs motifs
de rejet sont inclus dans le rapport de préparation.

## 3. Prédicteur supervisé d'activité Tox21

Le premier prédicteur couvre **NR-AR**, un essai rapporteur de signalisation
du récepteur aux androgènes. Une classe positive désigne une activité mesurée
dans cet essai *in vitro*; ce n'est ni une conclusion d'effet clinique ni une
classe universelle de toxicité. Une prédiction négative n'est pas une preuve
d'innocuité.

Le modèle est une forêt aléatoire scikit-learn sur empreintes Morgan/ECFP
RDKit (rayon 2, 2 048 bits). Les entrées invalides sont explicitement
comptabilisées et exclues; pour chaque endpoint, seules les lignes avec
`Dataset.w == 1` contribuent à l'apprentissage ou à l'évaluation. Les labels
masqués ne sont jamais convertis en négatifs. La sélection entre 100 et 300
arbres et le seuil de décision de Youden sont faits sur validation uniquement.
La partition test est utilisée par la commande distincte d'évaluation finale.

```powershell
python scripts/train_tox21_predictor.py --endpoint NR-AR
python scripts/evaluate_tox21_predictor.py --endpoint NR-AR
```

Les sorties du modèle, paramètres de prétraitement, classes comptées,
évaluation validation/test et prédictions CSV sont sous
`artifacts/tox21_predictors/nr_ar/`. L'évaluation rapporte ROC-AUC,
average-precision (aire PR), exactitude équilibrée, précision, rappel,
spécificité, F1, Brier et erreur de calibration attendue. Les intervalles à
95 % ROC-AUC, average-precision et Brier sont des bootstrap stratifiés sur
les classes (500 réplications, seed configurée). Le seuil est verrouillé sur
validation avant l'évaluation test.

### Résultats réellement mesurés — endpoint NR-AR

Exécution locale de `train_tox21_predictor.py` et de la commande d'évaluation
sur les partitions MoleculeNet scaffold :

| Partition | Positifs | Négatifs | Labels masqués | Mesures utilisées |
| --- | ---: | ---: | ---: | ---: |
| Train | 250 | 5 570 | 438 | 5 820 |
| Validation | 31 | 692 | 59 | 723 |
| Test | 27 | 688 | 68 | 715 |

La validation a retenu 300 arbres (ROC-AUC 0,7516) et un seuil de 0,3495.
Sur test, les résultats observés sont :

| Métrique | Résultat |
| --- | ---: |
| ROC-AUC (IC bootstrap 95 %) | 0,7301 (0,6163–0,8415) |
| Average precision (IC bootstrap 95 %) | 0,4434 (0,2592–0,6267) |
| Accuracy / balanced accuracy | 0,9566 / 0,7106 |
| Précision / rappel / spécificité / F1 | 0,4286 / 0,4444 / 0,9767 / 0,4364 |
| Brier (IC bootstrap 95 %) | 0,0459 (0,0400–0,0515) |
| ECE, 10 bins | 0,1253 |

Le test est très déséquilibré (27 positifs parmi 715 labels observés); une
accuracy élevée ne doit donc pas être interprétée seule. Le Brier et l'ECE
indiquent aussi que les probabilités ne doivent pas être considérées comme
parfaitement calibrées. Ces estimations concernent l'activité du modèle
rapporteur NR-AR seulement et ne démontrent ni une toxicité systémique ni une
innocuité. La distribution d'entraînement est limitée aux composés du Tox21;
la généralisation hors domaine reste inconnue.

## 4. Baseline MolGAN adversarial

Le baseline utilise sans remplacement le modèle `BasicMolGANModel` de DeepChem
2.8.0, qui hérite de `WGANModel`.

```text
bruit gaussien (dimension 10)
          │
          ▼
générateur MLP ──► distributions catégorielles softmax
                   nœuds (B,9,5) et arêtes (B,9,9,5)
                             │
graphe réel one-hot ─────────┤
                             ▼
                    discriminateur = critique WGAN
                    score scalaire, sans sigmoid
```

Pendant l'entraînement, le générateur transforme le bruit en distributions
différentiables pour les types de sommets et d'arêtes. Le discriminateur
MolGAN encode le graphe et renvoie un score scalaire de critique. L'échantillonnage
final utilise l'argmax bruité Gumbel implémenté par DeepChem. Les dimensions et
catégories restent celles de `MolGanFeaturizer` décrites plus haut.

Les objectifs vérifiés dans le code source DeepChem 2.8.0 ne sont pas une
binary cross-entropy :

- Générateur : minimiser la moyenne des scores du critique sur les graphes
  générés.
- Discriminateur/critique : minimiser
  `mean(score_réel - score_généré) + gradient_penalty`.
- La pénalité de gradient est celle de `WGANModel` (Gulrajani et al.), avec un
  coefficient configurable (10 par défaut). DeepChem optimise les réseaux via
  `fit_gan` et ses appels `fit_generator`; le code du projet ne réimplémente ni
  les pertes ni les mises à jour.

Il n'y a **pas de réseau de récompense de propriété dans le baseline** :
`property_reward` est absent (`null`) et aucun prédicteur Tox21 n'est appelé.
Le réseau de récompense de propriété du papier MolGAN nécessite une expérience
distincte; sa future intégration ne doit pas être confondue avec les pertes
WGAN adversariales.

Exécuter d'abord un smoke test CPU court, puis seulement si celui-ci réussit le
baseline complet :

```powershell
python scripts/train_baseline.py --smoke-test --device cpu
python scripts/train_baseline.py
```

Le smoke test utilise 16 molécules featurisées, un batch de 4 et une époque;
il valide le chemin d'optimisation et la génération, mais **ne démontre pas la
convergence**. Le baseline complet lit la configuration
`configs/default.json`. Un autre périphérique peut être demandé explicitement,
par exemple `--device cuda` si PyTorch a détecté CUDA.

Les seeds Python, NumPy et PyTorch sont configurées; les batches parcourent les
graphes dans un ordre déterministe. `--resume` demande à `fit_gan` de restaurer
le checkpoint DeepChem le plus récent. Un ajustement étroit de `restore` remet
à la bonne taille le tenseur `noise_input` dont DeepChem 2.8.0 sauvegarde la
forme du dernier batch; cette différence de forme provoquait sinon un échec de
chargement. DeepChem restaure les poids et l'étape globale. **Limitation de
reprise** : `fit_gan` demande des optimiseurs liés séparément aux paramètres du
générateur et du discriminateur, mais `TorchModel.save_checkpoint` 2.8.0 ne
sérialise que son optimizer principal. Une reprise restaure donc le modèle, mais
ne garantit pas la continuité exacte des états Adam propres à chaque réseau.
`--epochs` lors d'une reprise désigne le nombre d'époques supplémentaires.

Les artefacts sont enregistrés dans `artifacts/molgan_baseline/` :
checkpoints DeepChem, poids du générateur, configuration effective, métadonnées,
pertes réellement remontées par `fit_gan` et métriques d'échantillons à chaque
époque. Validité désigne ici une reconstruction et sanitation réussie par RDKit;
unicité déduplique les SMILES canoniques valides; nouveauté compare ces SMILES
à l'ensemble d'entraînement MolGAN seulement. Ces métriques ponctuelles ne
constituent pas une évaluation expérimentale ni une mesure de convergence.

Références techniques : De Cao et al., [MolGAN (2018)](https://arxiv.org/abs/1805.11973);
[BasicMolGANModel DeepChem 2.8.0](https://deepchem.readthedocs.io/en/2.8.0/api_reference/models.html);
Gulrajani et al., [Improved Training of Wasserstein GANs (2017)](https://arxiv.org/abs/1704.00028);
et le code installé `deepchem.models.torch_models.molgan` / `gan`.

## 5. Comparer MolGAN et les récompenses guidées Tox21

Entraîner d'abord le baseline et le prédicteur indépendant NR-AR :
```powershell
python scripts/train_baseline.py
python scripts/train_tox21_predictor.py --endpoint NR-AR
```

Puis comparer le baseline inchangé à deux fine-tunings REINFORCE :
```powershell
python scripts/compare_tox21_rewards.py --device cpu
```

Pour un test logiciel court, après avoir créé les deux artefacts requis :
```powershell
python scripts/compare_tox21_rewards.py --steps 1 --samples 8 --device cpu
```

Le générateur DeepChem renvoie des distributions catégorielles différentiables
pendant l'apprentissage; le sampling final est discret et ne transmet pas de
gradient. Le fine-tuning applique donc un estimateur de score REINFORCE :
log-probabilité des catégories d'atomes et de liaisons échantillonnées,
multipliée par une récompense détachée et centrée. Cette étape met à jour
uniquement le générateur; elle ne change ni la perte WGAN-GP ni le critique.

Le seul objectif biologique optimisé est **la baisse de probabilité prédite
d'activité dans l'essai rapporteur NR-AR**. Ce choix est une cible de calcul
spécifique à cet essai, pas l'affirmation que toute activité NR-AR est nocive,
ni un score général de toxicité ou de sécurité. L'incertitude est l'écart-type
des probabilités des arbres de la forêt, multiplié par deux (proxy de désaccord,
non calibré). Le support chimique est la similarité Tanimoto maximale aux
composés du train qui ont un label NR-AR observé. La diversité est la distance
Tanimoto moyenne aux autres molécules valides du batch. Ces quatre composantes
sont chacune dans `[0,1]`; leurs poids positifs sont normalisés par leur somme.
Une molécule invalide reçoit toujours `-1`; pour une molécule valide, la
récompense est :

```text
(w_activity * (1 - p_NR-AR)
 + w_diversity * batch_diversity
 + w_support * max_train_similarity
 - w_uncertainty * forest_disagreement) / sum(weights)
```

Les paramètres `tox21_guided_comparison.settings` de
[`configs/default.json`](./configs/default.json) comparent une configuration
centrée sur l'activité (poids 0,70) à une configuration plus équilibrée
(poids 0,40); diversité, support et désaccord sont aussi consignés séparément
à chaque pas. Les deux partent du même checkpoint générateur et utilisent la
même graine. Le modèle baseline reste conservé dans
`artifacts/molgan_baseline/`.

Pour chaque bras, le rapport inclut les graphes générés, la validité RDKit,
l'unicité des SMILES canoniques, la nouveauté par rapport aux structures train,
la distance Tanimoto moyenne entre uniques, la proportion d'activité NR-AR
prédite au seuil choisi sur validation, le désaccord des arbres et la part
hors support (`similarité maximale < 0,25`). Les sorties JSON, CSV de
composantes/résultats et grilles RDKit sont sous
`artifacts/tox21_guided_nr_ar/`. Les quantiles de probabilité décrivent les
échantillons générés; ils ne sont pas un intervalle de confiance statistique.
Le protocole ne consulte aucun label test pour l'optimisation. Des probabilités
basses peuvent provenir d'une exploitation du prédicteur ou d'un effondrement
de mode; vérifier le support, l'incertitude, diversité et validité ne remplace
pas un essai expérimental.

Des indicateurs de revue comparent aussi les changements d'activité prédite,
de support et de désaccord, et signalent les faibles taux d'unicité/distances
Tanimoto. Les seuils heuristiques sont sauvegardés dans la configuration; ces
signaux demandent une inspection et ne prouvent pas à eux seuls une exploitation
du prédicteur ou un effondrement de mode. Avec moins de deux uniques valides,
le rapport déclare ces conclusions non évaluables.

### Exécution smoke mesurée

La vérification logicielle CPU avec `--steps 1 --samples 8` a produit 8 graphes
par bras et **0 molécule RDKit valide** pour le baseline et les deux réglages.
Les deux bras guidés ont donc indiqué `policy_update_applied=0`: aucun graphe
valide n'a fourni de récompense différentielle, et les mises à jour bruitées
à récompense constante sont volontairement ignorées. Aucune activité NR-AR,
diversité, nouveauté ou comparaison de mode n'est estimable sur ces sorties.
Ce résultat confirme seulement l'exécution de bout en bout; il ne démontre
aucune amélioration. Le checkpoint disponible est celui du smoke test baseline,
pas un modèle MolGAN convergé. Il faut entraîner le baseline complet et
réexécuter la comparaison avant toute conclusion scientifique.

Le raccourci historique `python scripts/train.py` délègue maintenant à cette
comparaison; il ne réentraîne pas le baseline ni le prédicteur. Pour une
expérience reproductible, exécuter les deux commandes de prérequis ci-dessus.

Sorties principales dans `artifacts/` : générateur PyTorch, classifieurs Tox21,
molécules au format CSV, image RDKit et métriques JSON.

## 6. Réévaluer des molécules sauvegardées

```powershell
python scripts/evaluate.py
```

## Tests

```powershell
python -m pytest
```
