# 🏡 HousePricePredictor

Prédiction du prix de vente de maisons à partir de leurs caractéristiques, basée sur le dataset **Ames Housing** (Kaggle). Le projet compare six modèles de régression optimisés via **optimisation bayésienne (BayesSearchCV)** et expose les résultats dans une application **Streamlit** interactive.

---

## 📊 Résultats

| Modèle | R² | RMSE ($) | MAE ($) | CV Score |
|---|---|---|---|---|
| ElasticNet | 0.8800 | 33 489 | 19 489 | 0.829 |
| Lasso | 0.8752 | 33 797 | 19 637 | 0.827 |
| LightGBM | 0.8748 | 30 498 | 18 391 | 0.854 |
| **XGBoost** ✅ | **0.8686** | **26 992** | **16 983** | **0.850** |
| RandomForest | 0.8684 | 31 380 | 18 482 | 0.846 |
| Ridge | 0.8268 | 32 598 | 20 597 | 0.765 |

> **XGBoost** a été retenu comme modèle final : meilleur RMSE et MAE en dollars, avec un score CV cohérent avec le score test — signe d'une bonne capacité de généralisation.

---

## 📁 Structure du projet

```
HousePricePrediction/
├── data/
│   └── train.csv                  # Dataset Ames Housing
├── model/
│   ├── best_XGBoost.joblib        # Modèle final sauvegardé
│   ├── best_LightGBM.joblib
│   ├── best_Ridge.joblib
│   ├── best_Lasso.joblib
│   ├── best_ElasticNet.joblib
│   ├── best_RandomForest.joblib
│   ├── reference_input.csv        # Ligne de référence pour l'app
│   └── permutation_importance.csv
├── notebooks/
│   └── house_price_.ipynb         # Notebook principal
├── streamlit_apps/
│   └── house_app/
│       ├── app.py                 # Application Streamlit
│       ├── utils.py               # Fonctions utilitaires
│       └── requirements.txt
└── README.md
```

---

## 🔧 Pipeline

```
Données brutes (train.csv)
    ↓
Gestion des valeurs manquantes (Polars)
    ↓
Feature Engineering
    → AgeMaison, AgeRemod, AgeGarage, TotalSF, Qual_SF
    ↓
Sélection des variables
    → 12 numériques (corrélation) + 6 catégorielles (ANOVA)
    ↓
Preprocessing (ColumnTransformer)
    → StandardScaler (numériques)
    → OneHotEncoder (catégorielles)
    ↓
Optimisation bayésienne (BayesSearchCV, cv=5)
    ↓
Évaluation sur jeu de test (80/20, random_state=42)
    ↓
Sauvegarde Joblib + Application Streamlit
```

---

## ✨ Features créées

| Variable | Formule | Description |
|---|---|---|
| `AgeMaison` | `YrSold - YearBuilt` | Âge de la maison |
| `AgeRemod` | `YrSold - YearRemodAdd` (≥ 0) | Années depuis la dernière rénovation |
| `AgeGarage` | `YrSold - GarageYrBlt` (≥ 0) | Âge du garage |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Surface totale |
| `Qual_SF` | `OverallQual × GrLivArea` | Interaction qualité × surface |

---

## 🚀 Lancer l'application

### 1. Cloner le dépôt

```bash
git clone https://github.com/<your-username>/HousePricePrediction.git
cd HousePricePrediction
```

### 2. Installer les dépendances

```bash
pip install -r streamlit_apps/house_app/requirements.txt
```

### 3. Générer les modèles

Exécuter le notebook complet :

```bash
jupyter notebook notebooks/house_price_.ipynb
```

### 4. Lancer l'app

```bash
streamlit run streamlit_apps/house_app/app.py
```

---

## 🖥️ Application Streamlit

L'application expose 5 onglets :

- **🏠 Accueil** — présentation du projet et des facteurs clés du prix
- **💰 Estimer un bien** — formulaire de prédiction avec fourchette de confiance (±MAE), position sur le marché et importance des variables
- **📖 Mode d'emploi** — guide d'utilisation
- **📊 Analyse du marché** — comparaison des 6 modèles, distribution des prix du dataset
- **ℹ️ À propos** — stack technique, données, structure du projet

---

## 🛠️ Stack technique

| Outil | Usage |
|---|---|
| Python 3.10 | Langage principal |
| Polars | Manipulation des données |
| Pandas | Interopérabilité et stats |
| Scikit-Learn | Pipelines, métriques, prétraitement |
| XGBoost / LightGBM | Modèles gradient boosting |
| scikit-optimize | Optimisation bayésienne (BayesSearchCV) |
| Plotly | Visualisations interactives |
| Streamlit | Interface web |
| Joblib | Sérialisation des modèles |

---

## 📦 Données

- **Source** : [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Taille** : 1 460 observations, 81 variables
- **Cible** : `SalePrice` transformé en `log1p` pour la modélisation
- **Split** : 80% entraînement / 20% test — `random_state=42`
