import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

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

# Fonction de chargement du scaler (cachée)
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

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
scaler = load_scaler()

# Titre principal
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("Application de Machine Learning pour évaluer les demandes de prêt")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

with tab1:
    st.header("📊 Exploration des données")
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total demandes", f"{len(df):,}")
    
    with col2:
        approval_rate = (df['Loan_Status'] == 1).mean() * 100
        st.metric("✅ Taux d'approbation", f"{approval_rate:.1f}%")
    
    with col3:
        avg_loan = df['LoanAmount'].mean()
        st.metric("💰 Montant moyen", f"{avg_loan:,.0f} €")
    
    with col4:
        avg_income = df['CoapplicantIncome'].mean()
        st.metric("💵 Revenu moyen", f"{avg_income:,.0f} €")
    
    st.markdown("---")
    
    # Section Distributions
    st.subheader("📈 Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ApplicantIncome',
            title='Distribution des revenus des demandeurs',
            labels={'ApplicantIncome': 'Revenu (€)'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, y='LoanAmount',
            title='Distribution du montant des prêts',
            labels={'LoanAmount': 'Montant (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section Analyses
    st.subheader("🔍 Analyses")
    col1, col2 = st.columns(2)
    
    with col1:
        # Convertir 0/1 en texte pour le groupby
        df_temp = df.copy()
        df_temp['Loan_Status_Text'] = df_temp['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
        
        approval_by_edu = df_temp.groupby('Education')['Loan_Status'].mean() * 100
        fig = px.bar(
            x=approval_by_edu.index.map({1: 'Graduate', 0: 'Not Graduate'}),
            y=approval_by_edu.values,
            title='Taux d\'approbation par niveau d\'éducation',
            labels={'x': 'Éducation', 'y': 'Taux (%)'},
            color=approval_by_edu.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        loan_counts = df['Loan_Status'].value_counts()
        fig = px.pie(
            values=loan_counts.values,
            names=['Approved', 'Rejected'],
            title='Répartition des décisions',
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section Corrélations
    st.subheader("🔗 Corrélations")
    corr = df.select_dtypes(include=['number']).corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig.update_layout(title='Matrice de corrélation des variables numériques', height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset brut
    with st.expander("📄 Voir le dataset complet"):
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Faire une prédiction")
    st.write("Section à compléter")

with tab3:
    st.header("Performance du modèle")
    st.write("Section à compléter")

with st.form("prediction_form"):
    st.subheader("Informations du demandeur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        applicant_income = st.number_input("Revenu du demandeur (€)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Revenu du co-demandeur (€)", min_value=0, value=2000)
        loan_amount = st.number_input("Montant du prêt (€)", min_value=0, value=15000)
        loan_term = st.selectbox("Durée du prêt (mois)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
        credit_history = st.selectbox("Historique de crédit", [1, 0], format_func=lambda x: "Bon" if x == 1 else "Mauvais")

    with col2:
        education = st.selectbox("Niveau d'éducation", [1, 0], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
        married = st.selectbox("Marié", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
        dependents = st.selectbox("Nombre de personnes à charge", [0, 1, 2, 3])
        self_employed = st.selectbox("Travailleur indépendant", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
        property_area = st.selectbox("Zone de propriété", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])
    
    submitted = st.form_submit_button("Prédire l'approbation du prêt")

## Préparation des données pour la prédiction

# recréer les features engineerées
if submitted:
    # créer un dictionnaire avec les inputs
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Education': education,
        'Married': married,
        'Dependents': dependents,
        'Self_Employed': self_employed,
        'Property_Area': property_area
    }

    # créer un dataframe
    input_df = pd.DataFrame([input_data])

    #recréer les features engineerées
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['LoanAmountToIncome'] = input_df['LoanAmount'] / (input_df['Total_Income'] + 1)
    input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
    input_df['EMIToIncome'] = input_df['EMI'] / (input_df['Total_Income'] + 1)
    input_df['Log_LoanAmount'] = np.log(input_df['LoanAmount'] + 1)
    input_df['Log_Total_Income'] = np.log(input_df['Total_Income'] + 1)
    input_df['Has_Coapplicant'] = (input_df['CoapplicantIncome'] > 0).astype(int)

    # encoder les variables catégorielles
    input_df['Education'] = input_df['Education'].map({1: 'Graduate', 0: 'Not Graduate'})
    input_df['Married'] = input_df['Married'].map({1: 'Yes', 0: 'No'})
    input_df['Self_Employed'] = input_df['Self_Employed'].map({1: 'Yes', 0: 'No'})
    input_df['Property_Area'] = input_df['Property_Area'].map({0: 'Rural', 1: 'Semiurban', 2: 'Urban'})
    input_df = pd.get_dummies(input_df, columns=['Education', 'Married', 'Self_Employed', 'Property_Area'], drop_first=True)

## Faire la prédiction

#appeler le modèle

    # normaliser (si Logistic Regression)
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
    else:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

    # probabilité de la classe positive
    proba_approved = proba[1] * 100
    proba_rejected = proba[0] * 100

# afficher le résultat
    st.markdown("---")
    st.subheader("Résultat de la prédiction")
    if prediction == 1:
        st.success(" **Prêt approuvé** 🎉")
        st.metric("Probabilité d'approbation", f"{proba_approved:.1f}%")
    else:
        st.error(" **Prêt rejeté** ❌")
        st.metric("Probabilité de rejet", f"{proba_rejected:.1f}%")

    # barre de progression
    st.progress(proba_approved / 100)
    st.progress(proba_rejected / 100)

