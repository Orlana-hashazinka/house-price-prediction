
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils import load_model, load_reference, prepare_input, predict_price, load_market_data, get_market_stats
 
# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="HousePricePredictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# =========================
# CSS GLOBAL
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
body { font-family: 'Inter', sans-serif; }
 
.stApp { background: none; }
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background-image: url("https://images.unsplash.com/photo-1570129477492-45c003edd2be");
    background-size: cover;
    background-position: center;
    filter: blur(8px) brightness(0.6);
    z-index: -1;
}
 
.block-container {
    background-color: rgba(10, 20, 35, 0.78);
    backdrop-filter: blur(6px);
    padding: 3rem 2.5rem !important;
    max-width: 1100px;
    margin: 3rem auto;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}
 
.block-container h1 { color: #FFFFFF; font-size: 3rem; font-weight: 700; }
.block-container h2 { color: #FFFFFF; font-size: 1.8rem; font-weight: 400; }
.block-container h3 { color: #FFFFFF; font-size: 1.3rem; font-weight: 600; }
.block-container p, .block-container li, .block-container label, .block-container span {
    color: #E5E7EB; font-size: 0.95rem; line-height: 1.6;
}
 
div[data-baseweb="tab-list"] { justify-content: center !important; gap: 0.4rem; }
button[data-baseweb="tab"] {
    all: unset;
    background: rgba(0,0,0,0.6);
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.2);
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #1D4ED8;
    color: #E0EAFF;
    border-color: rgba(29,78,216,0.5);
}
 
.feature-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    text-align: center;
    transition: background 0.3s;
}
.feature-card:hover { background: rgba(255,255,255,0.1); }
.feature-card .icon { font-size: 2rem; margin-bottom: 8px; }
.feature-card .label { color: white; font-weight: 600; font-size: 0.95rem; }
.feature-card .desc { color: #9CA3AF; font-size: 0.82rem; margin-top: 4px; }
 
.price-box {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(29,78,216,0.3);
}
.price-box .label { color: #BFDBFE; font-size: 0.9rem; letter-spacing: 0.1em; text-transform: uppercase; }
.price-box .amount { color: #FFFFFF; font-size: 3rem; font-weight: 700; margin: 8px 0; }
.price-box .range { color: #93C5FD; font-size: 0.85rem; }
 
.bottom-bg {
    background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                      url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
    background-size: cover;
    background-position: center;
    border-radius: 18px;
    padding: 60px 20px;
    text-align: center;
    margin-top: 3rem;
    border: 1px solid rgba(255,255,255,0.08);
}
 
[data-testid="stMetricValue"] { color: white !important; }
[data-testid="stMetricLabel"] { color: #9CA3AF !important; }
</style>
""", unsafe_allow_html=True)
 
 
# =========================
# CHARGEMENT
# =========================
@st.cache_resource
def get_model():
    return load_model("XGBoost")
 
@st.cache_data
def get_reference():
    return load_reference()
 
@st.cache_data
def get_market():
    return load_market_data()
 
model     = get_model()
reference = get_reference()
market    = get_market()
stats     = get_market_stats(market)
 
 
# =========================
# TABS
# =========================
tabs = st.tabs([
    "🏠 Accueil",
    "💰 Estimer un bien",
    "📖 Mode d'emploi",
    "📊 Analyse du marché",
    "ℹ️ À propos"
])
 
 
# ======================================================
# TAB 0 — ACCUEIL
# ======================================================
with tabs[0]:
    st.markdown("""
        <h1 style="text-align:center; font-size:3.2rem; margin-bottom:0.3rem;">
            🏡 HousePricePredictor
        </h1>
        <p style="text-align:center; font-size:1.15rem; color:#9CA3AF; margin-top:0;">
            Estimez le prix de vente d'un bien immobilier grâce au Machine Learning.
        </p>
    """, unsafe_allow_html=True)
 
    st.markdown('<div style="margin-bottom:2rem;"></div>', unsafe_allow_html=True)
    st.markdown("---")
 
    st.markdown("### 🔍 Comment ça marche ?")
    st.markdown("""
    - **Renseignez** les caractéristiques de votre bien (surface, qualité, quartier...).
    - **Notre modèle XGBoost** analyse 18 variables clés pour estimer le prix.
    - **Obtenez** une estimation instantanée avec une fourchette de confiance et une analyse marché.
    """)
 
    st.markdown("---")
    st.markdown("### ✨ Les facteurs clés du prix")
 
    row1 = st.columns(3)
    row2 = st.columns(3)
 
    features_info = [
        ("🏗️", "Qualité générale", "Le facteur n°1 dans la prédiction du prix"),
        ("📐", "Surface habitable", "Plus le logement est grand, plus sa valeur augmente"),
        ("🚗", "Garage", "Capacité et état du garage influencent fortement le prix"),
        ("🏘️", "Quartier", "La localisation reste un critère décisif"),
        ("🛁", "Salles de bain", "Confort et valeur perçue du bien"),
        ("📅", "Âge & rénovation", "Ancienneté et travaux récents"),
    ]
 
    for col, (icon, label, desc) in zip(row1 + row2, features_info):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <div class="label">{label}</div>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.info("💡 Modèle entraîné sur le dataset Ames Housing (1 460 maisons) — XGBoost optimisé par BayesSearchCV. RMSE : 26 992 $ | MAE : 16 983 $")
 
    st.markdown("""
    <div class="bottom-bg">
        <h2 style="color:white; font-size:2rem; margin-bottom:10px;">Prêt à estimer votre bien ?</h2>
        <p style="color:#E5E7EB; font-size:1.1rem;">
            Rendez-vous dans l'onglet <strong>💰 Estimer un bien</strong> pour obtenir votre estimation en quelques secondes.
        </p>
    </div>
    """, unsafe_allow_html=True)
 
 
# ======================================================
# TAB 1 — ESTIMATION
# ======================================================
with tabs[1]:
    st.title("💰 Estimer le prix d'un bien")
    st.markdown("Renseignez les caractéristiques du bien pour obtenir une estimation.")
    st.markdown("---")
 
    col_left, col_right = st.columns(2)
 
    with col_left:
        st.markdown("#### 🏗️ Structure & Qualité")
        overall_qual  = st.slider("Qualité générale (1 = très mauvais, 10 = excellent)", 1, 10, 6)
        gr_liv_area   = st.number_input("Surface habitable (pi²)", 500, 6000, 1500)
        total_bsmt_sf = st.number_input("Surface sous-sol (pi²)", 0, 3000, 800)
        total_sf      = gr_liv_area + total_bsmt_sf
        qual_sf       = overall_qual * gr_liv_area
 
        st.markdown("#### 🚗 Garage")
        garage_cars   = st.selectbox("Capacité du garage (voitures)", [0, 1, 2, 3, 4], index=2)
        garage_area   = st.number_input("Surface garage (pi²)", 0, 1500, 480)
        garage_finish = st.selectbox("Finition du garage", ["NoGarage", "Unf", "RFn", "Fin"])
        garage_type   = st.selectbox("Type de garage", ["NoGarage", "Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types"])
 
    with col_right:
        st.markdown("#### 📍 Localisation")
        neighborhood = st.selectbox("Quartier (Neighborhood)", [
            "NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt","Gilbert",
            "Sawyer","NWAmes","SawyerW","Mitchel","BrkSide","Crawfor","IDOTRR",
            "Timber","NoRidge","StoneBr","SWISU","ClearCr","MeadowV","Blmngtn",
            "BrDale","Veenker","NPkVill","Blueste"
        ])
        ms_zoning      = st.selectbox("Zone résidentielle", ["RL", "RM", "FV", "RH", "C (all)"])
        sale_condition = st.selectbox("Condition de vente", ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"])
        foundation     = st.selectbox("Type de fondation", ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"])
 
        st.markdown("#### 📅 Âge & Rénovation")
        year_built = st.number_input("Année de construction", 1870, 2010, 1990)
        age_maison = 2010 - year_built
        age_remod  = st.number_input("Années depuis dernière rénovation", 0, 80, 10)
        age_garage = st.number_input("Âge du garage (années)", 0, 80, 15)
 
        st.markdown("#### 🛁 Équipements")
        full_bath = st.selectbox("Salles de bain complètes", [0, 1, 2, 3, 4], index=2)
 
    st.markdown("---")
 
    if st.button("🔮 Estimer le prix", type="primary"):
 
        user_inputs = {
            "Qual_SF":       qual_sf,
            "OverallQual":   overall_qual,
            "TotalSF":       total_sf,
            "GrLivArea":     gr_liv_area,
            "GarageCars":    garage_cars,
            "GarageArea":    garage_area,
            "TotalBsmtSF":   total_bsmt_sf,
            "FullBath":      full_bath,
            "AgeMaison":     age_maison,
            "YearBuilt":     year_built,
            "AgeRemod":      age_remod,
            "AgeGarage":     age_garage,
            "Neighborhood":  neighborhood,
            "GarageFinish":  garage_finish,
            "GarageType":    garage_type,
            "MSZoning":      ms_zoning,
            "Foundation":    foundation,
            "SaleCondition": sale_condition,
        }
 
        input_df = prepare_input(reference, user_inputs)
        price    = predict_price(model, input_df)
        marge    = 16983
        low, high = price - marge, price + marge
 
        # --- Prix ---
        st.markdown(f"""
        <div class="price-box">
            <div class="label">Estimation du prix de vente</div>
            <div class="amount">${price:,.0f}</div>
            <div class="range">Fourchette estimée : ${low:,.0f} — ${high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
 
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Surface habitable", f"{gr_liv_area} pi²")
        col_m2.metric("Qualité générale",  f"{overall_qual}/10")
        col_m3.metric("Quartier", neighborhood)
 
        # --- Analyse marché ---
        if stats:
            st.markdown("---")
            st.markdown("#### 📊 Position sur le marché")
 
            if price > stats["q75"]:
                st.warning("⚠️ Ce bien est dans le **quartile supérieur** du marché (top 25%)")
            elif price < stats["q25"]:
                st.info("💡 Ce bien est dans le **quartile inférieur** du marché (bottom 25%)")
            else:
                st.success("✅ Ce bien est dans la **fourchette médiane** du marché")
 
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Médiane marché",         f"${stats['median']:,.0f}")
            col_s2.metric("Votre bien vs médiane",  f"${price - stats['median']:+,.0f}")
            col_s3.metric("Fourchette marché Q1–Q3", f"${stats['q25']:,.0f} – ${stats['q75']:,.0f}")
 
            # Histogramme + position du bien
            fig_hist = px.histogram(
                market, x="SalePrice", nbins=50,
                title="Distribution des prix — Ames Housing",
                labels={"SalePrice": "Prix de vente ($)"},
                color_discrete_sequence=["#3B82F6"]
            )
            fig_hist.add_vline(
                x=price, line_dash="dash", line_color="#EF4444",
                annotation_text=f"Votre bien : ${price:,.0f}",
                annotation_font_color="#EF4444"
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
 
            # Comparaison avec la moyenne marché
            st.markdown("#### 📐 Votre bien vs la moyenne du marché")
            df_compare = pd.DataFrame({
                "Variable":       ["Surface (pi²)", "Garage (voitures)"],
                "Moyenne marché": [market["GrLivArea"].mean(), market["GarageCars"].mean()],
                "Votre bien":     [gr_liv_area, garage_cars]
            })
            df_melt = df_compare.melt(id_vars="Variable", var_name="Type", value_name="Valeur")
            fig_comp = px.bar(
                df_melt, x="Variable", y="Valeur", color="Type", barmode="group",
                color_discrete_map={"Moyenne marché": "#6B7280", "Votre bien": "#3B82F6"}
            )
            fig_comp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", legend_title_text=""
            )
            st.plotly_chart(fig_comp, use_container_width=True)
 
        # --- Importance des variables ---
        st.markdown("---")
        st.markdown("#### 🧠 Variables importantes (modèle XGBoost)")
        try:
            model_step    = model.named_steps["model"]
            importances   = model_step.feature_importances_
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            clean_names   = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
 
            df_imp = pd.DataFrame({
                "Feature":    clean_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(10)
 
            fig_imp = px.bar(
                df_imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Top 10 variables — importance native XGBoost"
            )
            fig_imp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", yaxis=dict(autorange="reversed"),
                showlegend=False, coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info(f"📌 Facteur principal : **{df_imp.iloc[0]['Feature']}**")
 
        except Exception:
            st.info("Importance des variables non disponible.")
 
 
# ======================================================
# TAB 2 — MODE D'EMPLOI
# ======================================================
with tabs[2]:
    st.title("📖 Comment utiliser l'application ?")
    st.markdown("---")
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 1️⃣ Renseignez le bien")
        st.markdown("""
        - **Structure** : qualité générale, surface habitable, sous-sol
        - **Garage** : capacité, surface, finition, type
        - **Localisation** : quartier, zone résidentielle, condition de vente
        - **Âge** : année de construction, rénovations, âge du garage
        """)
    with col_b:
        st.markdown("### 2️⃣ Obtenez l'estimation")
        st.markdown("""
        - Cliquez sur **🔮 Estimer le prix**
        - Le prix s'affiche avec une **fourchette de confiance** (±MAE du modèle)
        - Votre bien est **positionné sur le marché** grâce au train.csv
        - Les **variables les plus importantes** du modèle sont affichées
        """)
 
    st.markdown("---")
    st.markdown("### 🧠 Variables utilisées par le modèle")
    st.table({
        "Catégorie": ["Qualité", "Surface", "Garage", "Localisation", "Âge"],
        "Variables": [
            "OverallQual, Qual_SF",
            "GrLivArea, TotalSF, TotalBsmtSF",
            "GarageCars, GarageArea, GarageFinish, GarageType",
            "Neighborhood, MSZoning, Foundation, SaleCondition",
            "AgeMaison, YearBuilt, AgeRemod, AgeGarage"
        ]
    })
    st.info("💡 La fourchette correspond au MAE du modèle XGBoost sur les données test : **±16 983 $**")
 
 
# ======================================================
# TAB 3 — ANALYSE DU MARCHÉ
# ======================================================
with tabs[3]:
    st.title("📊 Analyse du marché immobilier")
    st.write("Résultats de la modélisation sur le dataset Ames Housing.")
    st.markdown("---")
 
    st.markdown("### 1. Performances des modèles")
    perf_df = pd.DataFrame({
        "Modèle":   ["ElasticNet", "Lasso", "LightGBM", "XGBoost", "RandomForest", "Ridge"],
        "R²":       [0.8800, 0.8752, 0.8748, 0.8686, 0.8684, 0.8268],
        "RMSE ($)": [33489, 33797, 30498, 26992, 31380, 32598],
        "MAE ($)":  [19489, 19637, 18391, 16983, 18482, 20597],
        "CV Score": [0.829, 0.827, 0.854, 0.850, 0.846, 0.765],
    })
 
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        fig = px.bar(
            perf_df, x="Modèle", y="MAE ($)",
            title="MAE en dollars par modèle (plus bas = meilleur)",
            color="MAE ($)", color_continuous_scale="Blues_r", text="MAE ($)"
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False, coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_t2:
        st.dataframe(perf_df.set_index("Modèle"), use_container_width=True)
 
    st.markdown("---")
    st.markdown("### 2. Pourquoi XGBoost a été retenu")
    col_x1, col_x2, col_x3 = st.columns(3)
    col_x1.metric("MAE ($)",   "16 983 $", "Meilleur de tous les modèles")
    col_x2.metric("RMSE ($)",  "26 992 $", "Meilleur de tous les modèles")
    col_x3.metric("CV Score",  "0.850",    "Cohérent avec le score test")
 
    st.success("""
    **Pourquoi pas ElasticNet malgré son R² de 0.88 ?**
    Son score CV (0.829) est bien inférieur à son R² test → instabilité de généralisation.
    XGBoost présente des scores cohérents entre CV et test, et minimise les erreurs en dollars.
    """)
 
    if market is not None:
        st.markdown("---")
        st.markdown("### 3. Distribution des prix — Ames Housing")
        fig_m = px.histogram(
            market, x="SalePrice", nbins=50,
            title="Distribution des prix de vente",
            color_discrete_sequence=["#3B82F6"]
        )
        fig_m.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_m, use_container_width=True)
 
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Prix médian", f"${market['SalePrice'].median():,.0f}")
        col_d2.metric("Prix moyen",  f"${market['SalePrice'].mean():,.0f}")
        col_d3.metric("Écart-type",  f"${market['SalePrice'].std():,.0f}")
 
 
# ======================================================
# TAB 4 — À PROPOS
# ======================================================
with tabs[4]:
    st.title("ℹ️ À propos du projet")
 
    st.markdown("""
    ### 📖 Présentation
    **HousePricePredictor** est un projet de prédiction de prix immobiliers basé sur le  
    dataset **Ames Housing** (Kaggle). Il compare six modèles de régression optimisés  
    via **optimisation bayésienne (BayesSearchCV)**.
 
    ---
    """)
 
    st.markdown("### 🛠️ Technologies utilisées")
    st.markdown("""
    - **Python 3.10** — Langage principal
    - **Streamlit** — Interface interactive
    - **Polars & Pandas** — Manipulation des données
    - **Scikit-Learn** — Pipelines, prétraitement, métriques
    - **XGBoost / LightGBM** — Modèles gradient boosting
    - **scikit-optimize (skopt)** — Optimisation bayésienne
    - **Plotly** — Visualisations interactives
    - **Joblib** — Sérialisation des modèles
    """)
 
    st.markdown("---")
    st.markdown("### 📊 Données")
    st.markdown("""
    - **Source** : Ames Housing Dataset — Iowa State University / Kaggle
    - **Taille** : 1 460 observations, 81 variables
    - **Cible** : `SalePrice` (transformé en `log1p` pour la modélisation)
    - **Split** : 80% entraînement / 20% test — `random_state=42`
    """)
 
    st.markdown("---")
    st.markdown("### 📁 Structure du projet")
    st.code("""
 HousePricePrediction/
├── data/
│   └── train.csv
├── model/
│   ├── best_XGBoost.joblib
│   ├── best_LightGBM.joblib
│   ├── best_Ridge.joblib
│   └── reference_input.csv
├── notebooks/
│   └── house_price_.ipynb
├── streamlit_apps/
│   └── house_app/
│       ├── app.py
│       ├── utils.py
│       └── requirements.txt
""", language="bash")

    st.markdown("---")
    st.markdown("### ▶️ Lancer l'application")
    st.code("streamlit run streamlit_app/app.py", language="bash")