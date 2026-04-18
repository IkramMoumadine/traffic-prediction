````markdown
# Traffic Prediction — PEMS07

Prédiction du flux de trafic routier sur le dataset **PEMS07** (pas de 5 minutes, réseau autoroutier de Los Angeles).

## Modèles implémentés

| Modèle      |          Script         |                  Description                |
|-------------|-------------------------|---------------------------------------------|
| CNN 1D      | mains/main_cnn1d.py     | Convolutions temporelles unidimensionnelles |
| CNN 2D      | mains/main_cnn2d.py     | Convolutions spatio-temporelles             |
| GRU         | mains/main_gru.py       | Réseau récurrent multivarié (GRU)           |
| LSTM        | mains/main_lstm.py      | Réseau récurrent multivarié (LSTM)          |
| GCN         | mains/main_gcn.py       | Graph Convolutional Network                 |
| ARIMA       | mains/main_arima.py     | Modèle statistique de référence             |
| CNN + ARIMA | mains/main_cnn_arima.py | Hybride CNN + correction ARIMA des résidus  |
| GCN + LSTM  | mains/main_gcn_lstm.py  | Hybride spatio-temporel GCN-LSTM            |

---

## Installation

### 1. Prérequis
- Python 3.10+
- CPU suffisant (GPU optionnel)

### 2. Environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
````

### 3. Dépendances

```bash
pip install -r requirements.txt
```

---

## Dataset (obligatoire)

Le fichier `PEMS07.npz` n’est **pas inclus** dans le dépôt (fichier volumineux).

Télécharger le dataset depuis la source officielle :
**Caltrans Performance Measurement System (PeMS)** — jeu **PEMSD7**

Placer les fichiers dans :

```
data/PEMS07/
├── PEMS07.npz
└── PeMSD7_M_Station_Info.csv
```

---

## Lancement

```bash
# Un seul modèle
python run_all.py --models cnn1d

# Plusieurs modèles
python run_all.py --models cnn1d gru gcn

# Tous les modèles
python run_all.py --models all
```

---

## Visualisation cartographique

```bash
python capteurs_cartes.py
```

Génère le fichier : `results/capteurs_carte_pems07.html`

---

## Structure du projet

```
traffic-prediction/
├── config.py
├── run_all.py
├── capteurs_cartes.py
├── requirements.txt
├── data/
│   └── PEMS07/              # dataset à télécharger
├── mains/                   # scripts d'exécution par modèle
├── src/                     # code source organisé
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
└── results/                 # généré automatiquement (non versionné)
```

---

## Résultats

Toutes les figures, courbes d’apprentissage et modèles sont générés automatiquement dans :

```
results/<modèle>/
```

Ce dossier n’est pas présent sur GitHub : il est recréé à l’exécution.

---

## Reproductibilité

Après clonage du dépôt :

```bash
pip install -r requirements.txt
python run_all.py --models all
```

Toutes les figures et modèles sont régénérés automatiquement.

```
```
