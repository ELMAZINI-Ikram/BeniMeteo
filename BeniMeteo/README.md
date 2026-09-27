# BeniMeteo — Surveillance climatique & alertes de vulnérabilité

**Plateforme d'aide à la décision qui croise données météorologiques, prévisions par Machine Learning et données démographiques pour identifier les zones prioritaires face aux risques climatiques** dans la région Béni Mellal-Khénifra (Maroc).

![Python](https://img.shields.io/badge/Python-3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![SARIMA](https://img.shields.io/badge/S%C3%A9ries%20temporelles-SARIMA-purple)
![QGIS](https://img.shields.io/badge/SIG-QGIS-green)

> 🇬🇧 *Climate monitoring and early-warning platform for the Béni Mellal-Khénifra region (Morocco): weather data pipeline, time-series forecasting (SARIMA, XGBoost), vulnerability scoring combining weather and demographic data at village level, and GIS export for QGIS.*

Projet réalisé en équipe dans le cadre de la compétition **The Moroccan TSP Competition**.

---

## Sommaire

- [Contexte et problématique](#contexte-et-problématique)
- [Solution](#solution)
- [Aperçu](#aperçu)
- [Données](#données)
- [Modèles de prédiction](#modèles-de-prédiction)
- [Score de vulnérabilité et alertes](#score-de-vulnérabilité-et-alertes)
- [Intégration SIG (QGIS)](#intégration-sig-qgis)
- [Architecture](#architecture)
- [Installation](#installation)
- [Compétences mobilisées](#compétences-mobilisées)
- [Équipe](#équipe)

---

## Contexte et problématique

La région Béni Mellal-Khénifra est exposée à des aléas climatiques variés : vagues de chaleur en plaine, gel et froid dans les zones d'altitude (Azilal, Khénifra), fortes précipitations et risques d'inondation. L'impact de ces aléas dépend fortement de la population exposée : enfants, personnes âgées, accès limité aux soins.

**Enjeu :** ne pas se limiter à prévoir la météo, mais identifier **où** et **pour qui** le risque est le plus élevé, afin de prioriser les alertes.

## Solution

```
Données météo (Open-Meteo) ─► Nettoyage & variables dérivées ─► Prévision (SARIMA, XGBoost)
        │
Données démographiques HCP ─► Score de vulnérabilité par douar ─► Alertes par zone
        │
        └─► Tableau de bord interactif (Streamlit) + Export GeoJSON vers QGIS
```

## Aperçu

<!-- Ajouter les captures d'écran dans un dossier docs/ puis décommenter :
![Tableau de bord](docs/dashboard.png)
![Carte des alertes](docs/carte_alertes.png)
![Prévisions](docs/previsions.png)
-->

*Captures d'écran à venir.*

## Données

| Source | Contenu | Accès |
|--------|---------|-------|
| [Open-Meteo](https://open-meteo.com) | Historique météo depuis 2010 et prévisions à 16 jours | API gratuite, sans clé |
| HCP Maroc | Données démographiques de **2 485 douars** | Fichier CSV local |

**Variables météorologiques exploitées :** températures (max, min, moyenne, indice de chaleur), précipitations (journalières, cumuls sur 7 et 30 jours), vent (vitesse max, rafales, direction), humidité, UV, ensoleillement et rayonnement, pression atmosphérique, évapotranspiration, enneigement.

**Variables démographiques par douar :**

| Champ | Description |
|-------|-------------|
| `population` | Population totale |
| `age_m15` | Part de la population de moins de 15 ans |
| `age_15_59` | Part de la population de 15 à 59 ans |
| `age_60_plus` | Part des personnes de 60 ans et plus |

## Modèles de prédiction

| Modèle | Type | Intérêt |
|--------|------|---------|
| **SARIMA** | Statistique, séries temporelles saisonnières | Capture la saisonnalité propre aux séries climatiques |
| **XGBoost** | Machine Learning multi-variables | Exploite les retards temporels (lags) et les variables explicatives (humidité, vent…) |

Les performances des modèles sont évaluées et comparées à l'aide de métriques dédiées (module `evaluation.py`).

## Score de vulnérabilité et alertes

Le score de vulnérabilité (0 à 100) combine l'intensité de l'aléa météorologique et la sensibilité de la population exposée.

**Facteurs pris en compte :** canicule et forte chaleur, humidex dangereux, risque d'inondation, vents violents, UV extrême, gel et froid en zones d'altitude, accès limité aux soins.

| Niveau | Score | Signification |
|--------|:-----:|---------------|
| 🟢 Vert | 0 – 14 | Pas d'alerte |
| 🟡 Jaune | 15 – 34 | Vigilance |
| 🟠 Orange | 35 – 59 | Alerte |
| 🔴 Rouge | 60 – 100 | Alerte critique |

## Intégration SIG (QGIS)

1. Exporter les zones au format **GeoJSON** depuis l'onglet *Export QGIS* du tableau de bord
2. Dans QGIS : *Couche → Ajouter une couche → Ajouter une couche vecteur*
3. Système de coordonnées : **EPSG:4326** (WGS 84) ou **EPSG:26191** (Merchich / Nord Maroc)
4. Un script **PyQGIS** est fourni pour automatiser l'import et la symbologie

## Architecture

```
BeniMeteo/
├── dashboard.py              # Application Streamlit
├── requirements.txt
├── data/
│   └── hcp_douar_bnk.csv     # Données démographiques HCP (2 485 douars)
└── src/
    ├── data_loader.py        # Collecte via l'API Open-Meteo (historique + prévisions)
    ├── preprocessing.py      # Nettoyage et variables dérivées
    ├── hcp_loader.py         # Chargement des données HCP
    ├── alerts.py             # Score de vulnérabilité et alertes par zone
    ├── evaluation.py         # Métriques d'évaluation des modèles
    ├── visualization.py      # Graphiques Plotly
    └── models/
        ├── sarima.py         # Modèle SARIMA
        └── xgboost_model.py  # Modèle XGBoost
```

**Stack :** Python · pandas · XGBoost · SARIMA · Plotly · Streamlit · QGIS / PyQGIS

## Installation

```bash
git clone https://github.com/ELMAZINI-Ikram/BeniMeteo.git
cd BeniMeteo

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
streamlit run dashboard.py
```

Le tableau de bord s'ouvre sur **http://localhost:8501**. Aucune clé d'API n'est nécessaire.

## Compétences mobilisées

| Domaine | Compétences |
|---------|-------------|
| **Data engineering** | Collecte de données via API, nettoyage, création de variables dérivées |
| **Séries temporelles & ML** | Prévision SARIMA et XGBoost, feature engineering (lags), évaluation de modèles |
| **Analyse de risques** | Construction d'un score de vulnérabilité combinant aléa et exposition |
| **SIG** | Export GeoJSON, systèmes de coordonnées, automatisation PyQGIS |
| **Data visualisation** | Tableau de bord interactif Streamlit et Plotly |
| **Travail en équipe** | Projet collaboratif réalisé en contexte de compétition |

## Équipe

Projet réalisé par trois élèves ingénieur(e)s en Transformation Digitale Industrielle — ENSA Béni Mellal :

- **Ikram ELMAZINI** — [@ELMAZINI-Ikram](https://github.com/ELMAZINI-Ikram)
- **[Prénom NOM]** — [@pseudo-github](https://github.com/pseudo-github)
- **[Prénom NOM]** — [@pseudo-github](https://github.com/pseudo-github)
