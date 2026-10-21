# Environnement reproductible

## Versions retenues

Utiliser **CPython 3.11 x64**. L’ordinateur vérifié est sous Windows 11 x64 ;
Python 3.11.2 et Anaconda Python 3.9.13 y sont disponibles. DeepChem 2.8.0
publie des classifiers jusqu’à Python 3.11 et `Requires-Python <3.12`, ce qui
exclut Python 3.12 et plus pour cette version. Python 3.11 est donc le choix le
plus récent vérifié parmi les interpréteurs locaux.

Les dépendances directes sont fixées dans `requirements.txt` et `pyproject.toml` :

| Paquet | Version |
| --- | ---: |
| DeepChem | 2.8.0 |
| PyTorch | 2.5.1 |
| RDKit | 2024.3.5 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| SciPy | 1.14.1 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.9.2 |
| pytest | 8.3.3 |

Ces versions sont des versions stables. NumPy 1.26 est retenu pour éviter les
problèmes de compatibilité avec les bibliothèques scientifiques binaires plus
anciennes utilisées par DeepChem 2.8.0. L’installation de DeepChem n’utilise
pas l’extra `deepchem[torch]` : il inclut des composants optionnels sans lien
avec ce projet, tandis que le backend MolGAN requis est fourni par PyTorch.

## Windows PowerShell (CUDA)

Sur la machine vérifiée, `nvidia-smi` indique une NVIDIA GeForce RTX 3050 4 Go
et un pilote 576.52 (CUDA 12.9). Le wheel PyTorch CUDA 12.1 pour CPython 3.11
Windows est publié par l’index officiel PyTorch et convient à ce pilote.
L’environnement est créé dans `.venv-m1`, sans installer ni modifier les
paquets du Python global ni le `.venv` existant.

Depuis la racine du projet :

```powershell
$Python311 = "C:\Users\bouac\AppData\Local\Programs\Python\Python311\python.exe"
& $Python311 -m venv .venv-m1
.\.venv-m1\Scripts\Activate.ps1
python -m pip install pip==24.3.1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/diagnose_environment.py
python -m pytest
```

Si l’activation PowerShell est bloquée, ne changez pas la stratégie d’exécution
globale : appelez directement `.venv-m1\Scripts\python.exe` à la place de
`python`. Pour CPU seulement, remplacez `cu121` par `cpu` dans l’URL de l’index.

## Google Colab

Sélectionnez un runtime Python 3.10 ou 3.11. DeepChem 2.8.0 ne prend pas en
charge Python 3.12+. Dans une cellule Colab :

```python
!python --version
!python -m pip install pip==24.3.1
!python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
!python -m pip install -r requirements.txt
!python scripts/diagnose_environment.py
!python -m pytest
```

Après remplacement du PyTorch préinstallé dans Colab, redémarrez la session si
la plateforme le demande, puis exécutez le diagnostic et les tests. Pour un
runtime CPU, utiliser l’index `https://download.pytorch.org/whl/cpu`.

## Diagnostic et limites de plateforme

`python scripts/diagnose_environment.py` rapporte le système, l’interpréteur,
les versions de paquets, les imports requis, la version CUDA liée à PyTorch et
les caractéristiques des GPU accessibles. Le code de sortie est non nul si un
import obligatoire échoue. `python scripts/diagnose_environment.py --json`
produit le même rapport en JSON.

Le wheel CUDA PyTorch fournit son runtime CUDA ; un toolkit CUDA séparé n’est
pas nécessaire pour l’exécuter. Un pilote NVIDIA compatible reste requis.
Les 4 Go de VRAM de la RTX 3050 limitent les tailles de batch et les
entraînements longs ; Google Colab avec GPU est une alternative. Le support JAX
de DeepChem n’est pas officiellement disponible sous Windows ; ce projet
n’utilise pas JAX. Les modèles et métriques du projet sont pédagogiques :
un risque toxique prédit ne constitue ni un avis médical ni une validation
toxicologique expérimentale.

## Sources officielles consultées

- [DeepChem 2.8.0 sur PyPI (Python requis et classifiers)](https://pypi.org/project/deepchem/2.8.0/)
- [Installation DeepChem (backend PyTorch et note Windows/JAX)](https://deepchem.readthedocs.io/en/latest/get_started/installation.html)
- [Versions antérieures et commandes d’installation PyTorch](https://pytorch.org/get-started/previous-versions/)
- [Index officiel des wheels PyTorch CUDA 12.1](https://download.pytorch.org/whl/cu121/torch/)
- [Wheels RDKit et plateformes](https://pypi.org/project/rdkit/2024.3.5/)
- [Installation pandas 2.2.3](https://pandas.pydata.org/pandas-docs/version/2.2.3/getting_started/install.html)
- [Installation scikit-learn 1.5](https://scikit-learn.org/1.5/install.html)
- [Installation matplotlib 3.9.2](https://matplotlib.org/3.9.2/install/index.html)
- [Guide pytest 8.3](https://docs.pytest.org/en/8.3.x/getting-started.html)
- [Environnements virtuels Python 3.11](https://docs.python.org/3.11/library/venv.html)
