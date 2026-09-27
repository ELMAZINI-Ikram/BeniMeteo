# 🌦️ BeniMeteo — Béni Mellal-Khénifra

Système de surveillance climatique, prédiction ML et alertes vulnérabilité
pour la région **Béni Mellal-Khénifra**, Maroc.

## 🚀 Installation et lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le dashboard
streamlit run dashboard.py
```

## 📁 Structure du projet

```
BeniMeteo/
├── dashboard.py              # Application Streamlit principale
├── requirements.txt
├── data/
│   └── hcp_douar_bnk.csv    # Données démographiques HCP (2485 douars)
└── src/
    ├── data_loader.py        # API Open-Meteo (historique + prévisions)
    ├── preprocessing.py      # Nettoyage + variables dérivées
    ├── alerts.py             # Calcul vulnérabilité + alertes zones
    ├── hcp_loader.py         # Chargement données HCP Maroc
    ├── visualization.py      # Graphiques Plotly
    ├── evaluation.py         # Métriques modèles ML
    └── models/
        ├── sarima.py         # Modèle SARIMA
        └── xgboost_model.py  # Modèle XGBoost
```

## 🌐 Sources de données

| Source | Description | API |
|--------|-------------|-----|
| [Open-Meteo](https://open-meteo.com) | Données météo historiques (2010→) + prévisions 16j | Gratuite, sans clé |
| HCP Maroc | Données démographiques par douar (fichier CSV) | Fichier local |

### Variables météo disponibles
- 🌡️ Températures (max, min, moyenne, indice chaleur)
- 🌧️ Précipitations (journalière, cumulée 7j/30j)
- 💨 Vent (vitesse max, rafales, direction)
- 💧 Humidité (max, min, moyenne)
- ☀️ UV, ensoleillement, rayonnement
- 📊 Pression atmosphérique, évapotranspiration
- ❄️ Enneigement

## 🗺️ Intégration QGIS

1. Exporter les zones en GeoJSON depuis l'onglet **Export QGIS**
2. Importer dans QGIS : `Couche > Ajouter couche vecteur`
3. Projection : **EPSG:4326** (WGS84) ou **EPSG:26191** (Lambert Maroc)
4. Utiliser le script **PyQGIS** fourni pour automatiser

## 🔮 Modèles de prédiction

- **SARIMA** : modèle statistique saisonnier, idéal pour séries climatiques
- **XGBoost** : machine learning multi-features (lags, humidité, vent...)

## 👥 Données démographiques HCP

Le fichier `hcp_douar_bnk.csv` contient 2485 douars avec :
- `population` : population totale
- `age_m15` : % population < 15 ans
- `age_15_59` : % population active
- `age_60_plus` : % personnes âgées

Le score de vulnérabilité combine ces données avec les conditions météo
pour identifier les zones prioritaires pour les alertes.

## ⚠️ Système d'alertes

| Niveau | Score | Couleur |
|--------|-------|---------|
| 🟢 VERT | 0–14 | Pas d'alerte |
| 🟡 JAUNE | 15–34 | Vigilance |
| 🟠 ORANGE | 35–59 | Alerte |
| 🔴 ROUGE | 60–100 | Alerte critique |

Les alertes prennent en compte :
- Canicule / forte chaleur
- Humidex dangereux
- Risque d'inondation
- Vents violents
- UV extrême
- Gel / froid (zones d'altitude : Azilal, Khénifra)
- Accès aux soins limité
