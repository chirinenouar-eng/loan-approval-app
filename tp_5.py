import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction de chargement des données (cachée)
@st.cache_data
def load_data():
    return pd.read_csv("loan_data_clean.csv")

# Fonction de chargement du modèle (cachée)
@st.cache_resource
def load_model(model_name):
    if model_name == "Logistic Regression":
        return joblib.load("logistic_regression.pkl")
    else:
        return joblib.load("random_forest.pkl")

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Logistic Regression", "Random Forest"]
)

# Charger les données et le modèle
df = load_data()
model = load_model(model_choice)

# Titre principal
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("Application de Machine Learning pour évaluer les demandes de prêt")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

with tab1:
    st.header("Exploration des données")
    st.dataframe(df, use_container_width=True)
    st.caption(f"Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")

with tab2:
    st.header("Faire une prédiction")
    st.write("Section à compléter")

with tab3:
    st.header("Performance du modèle")
    st.write("Section à compléter")

## visualisations essentielles avec plotly

# histogramme (distribution)
fig = px.histogram(
    df, 
    x="LoanAmount", 
    title="Distribution du montant des prêts",
    labels={"LoanAmount": "Montant(€)"},
    color_discrete_sequence=["#636EFA"]
)
st.plotly_chart(fig, use_container_width=True)

# boxplot (outliers)
fig2 = px.box(
    df, 
    y="ApplicantIncome", 
    title="Distribution des Revenus des Demandeurs (Boxplot)",
    labels={"ApplicantIncome": "Revenu (€)"},
)
st.plotly_chart(fig2)

# bar chart (comparaison)
approval_by_education = df.groupby("Education")["Loan_Status"].mean() * 100

fig3 = px.bar(
    x=approval_by_education.index,
    y=approval_by_education.values,
    title="Taux d\'approbation des prêts par niveau d\'éducation",
    labels={"x": "Niveau d'éducation", "y": "Taux d'approbation (%)"},
    color=approval_by_education.values,
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig3)

# scatter plot
fig4 = px.scatter(
    df, 
    x="ApplicantIncome", 
    y="LoanAmount", 
    color="Loan_Status",
    title="REelation revenu / Montant du prêt",
    labels={"TotalIncome": "Revenu total (€)", "LoanAmount": "Montant(€)"},
    hover_data=["Credit_History"]
)
st.plotly_chart(fig4)

# corrélation heatmap
corr = df.select_dtypes(include=['number']).corr()

fig5 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu',
    zmid=0
))
fig5.update_layout(title="Matrice de corrélation")
st.plotly_chart(fig5)

# pie chart (proportions)
loan_status_counts = df["Loan_Status"].value_counts()

fig6 = px.pie(
    values=loan_status_counts.values,
    names=['Approved', 'Rejected'],
    title="Répartition des statuts de prêt",
    color_discrete_sequence=["#00CC96", "#EF553B"]
)
st.plotly_chart(fig6)

## Dashboard multi-colonnes

# layout professionnel avec métriques et graphiques
# métriques en haut
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total demandes", len(df))

with col2:
    approval_rate = df["Loan_Status"].mean() * 100
    st.metric("Taux d'approbation", f"{approval_rate:.1f}%")

with col3:
    avg_loan = df["LoanAmount"].mean()
    st.metric("Montant moyen", f"{avg_loan:.0f} €")

with col4:
    avg_income = df["TotalIncome"].mean()
    st.metric("Moyenne des revenus", f"{avg_income:.0f} €")
st.markdown("---")

# graphiques en dessous
col1, col2 = st.columns(2)

with col1:
    fig7 = px.histogram(df, x="ApplicantIncome", title="Distribution des Revenus des Demandeurs")
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    fig8 = px.box(df, y="LoanAmount", title="Distribution des Montants des Prêts")
    st.plotly_chart(fig8, use_container_width=True)