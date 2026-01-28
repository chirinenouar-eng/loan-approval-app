import streamlit as st
import pandas as pd
import numpy as np
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
    
@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except:
        return None

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
    st.header("🤖 Faire une prédiction")
    st.write("Remplissez les informations ci-dessous pour obtenir une prédiction d'approbation de prêt.")
    
    # =========================================================================
    # FORMULAIRE DE SAISIE
    # =========================================================================
    
    # Utiliser st.form pour grouper les inputs et éviter les réexécutions
    with st.form("prediction_form"):
        
        # Créer 2 colonnes pour organiser le formulaire
        col1, col2 = st.columns(2)
        
        # ---------------------------------------------------------------------
        # COLONNE 1 : INFORMATIONS FINANCIÈRES
        # ---------------------------------------------------------------------
        with col1:
            st.subheader("💰 Informations financières")
            
            gender = st.selectbox(
                "Genre",
                options=[1, 0],
                format_func=lambda x: "👨 Homme" if x == 1 else "👩 Femme",
                help="Genre du demandeur"
            )
            # Revenu du demandeur
            applicant_income = st.number_input(
                "Revenu mensuel du demandeur (€)",
                min_value=0,
                max_value=100000,
                value=5000,
                step=100,
                help="Revenu mensuel net du demandeur principal"
            )
            
            # Revenu du co-demandeur
            coapplicant_income = st.number_input(
                "Revenu mensuel du co-demandeur (€)",
                min_value=0,
                max_value=100000,
                value=0,
                step=100,
                help="Revenu mensuel net du co-demandeur (mettre 0 si pas de co-demandeur)"
            )
            
            # Montant du prêt
            loan_amount = st.number_input(
                "Montant du prêt demandé (€)",
                min_value=1000,
                max_value=1000000,
                value=150000,
                step=1000,
                help="Montant total du prêt demandé"
            )
            
            # Durée du prêt
            loan_term = st.number_input(
                "Durée du prêt (mois)",
                min_value=12,
                max_value=480,
                value=360,
                step=12,
                help="Durée de remboursement en mois (ex: 360 mois = 30 ans)"
            )
        
        # ---------------------------------------------------------------------
        # COLONNE 2 : INFORMATIONS PERSONNELLES
        # ---------------------------------------------------------------------
        with col2:
            st.subheader("👤 Informations personnelles")
            
            # Historique de crédit
            credit_history = st.selectbox(
                "Historique de crédit",
                options=[1, 0],
                format_func=lambda x: "✅ Bon historique" if x == 1 else "❌ Mauvais historique",
                help="Indique si le demandeur a un bon historique de crédit"
            )
            
            # Niveau d'éducation
            education = st.selectbox(
                "Niveau d'éducation",
                options=[1, 0],
                format_func=lambda x: "🎓 Graduate" if x == 1 else "📚 Not Graduate",
                help="Niveau d'études du demandeur"
            )
            
            # Statut marital
            married = st.selectbox(
                "Statut marital",
                options=[1, 0],
                format_func=lambda x: "💑 Marié(e)" if x == 1 else "🧍 Célibataire",
                help="Statut marital du demandeur"
            )
            
            # Personnes à charge
            dependents = st.number_input(
                "Nombre de personnes à charge",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Nombre de personnes financièrement dépendantes du demandeur"
            )
            
            # Travailleur indépendant
            self_employed = st.selectbox(
                "Travailleur indépendant",
                options=[0, 1],
                format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
                help="Indique si le demandeur est travailleur indépendant"
            )
            
            # Zone du bien
            property_area = st.selectbox(
                "Zone du bien immobilier",
                options=["Urban", "Semiurban", "Rural"],
                help="Type de zone où se situe le bien"
            )
        
        # Bouton de soumission
        st.markdown("---")
        submitted = st.form_submit_button(
            "🔮 Prédire l'approbation du prêt",
            use_container_width=True,
            type="primary"
        )
    
    # =========================================================================
    # TRAITEMENT DE LA PRÉDICTION
    # =========================================================================
    
    if submitted:
        # Vérifications de cohérence
        st.markdown("---")
        
        # Afficher un spinner pendant le traitement
        with st.spinner('Analyse en cours...'):
            
            # -----------------------------------------------------------------
            # ÉTAPE 1 : CRÉER LE DATAFRAME D'INPUT
            # -----------------------------------------------------------------
            
            # TODO : Créer un dictionnaire avec toutes les features de base
            input_data = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Education': education,
                'Gender_Male': gender,
                'Married_Yes': married,
                'Dependents': dependents,
                'SelfEmployed_Yes': self_employed,
                # One-hot encoding pour Property_Area
                'Area_Semiurban': 1 if property_area == "Semiurban" else 0,
                'Area_Urban': 1 if property_area == "Urban" else 0
            }
            
            # Créer un DataFrame
            input_df = pd.DataFrame([input_data])
            
            # -----------------------------------------------------------------
            # ÉTAPE 2 : FEATURE ENGINEERING
            # -----------------------------------------------------------------
            
            # TODO : Recréer EXACTEMENT les mêmes features qu'à l'entraînement
            
            # Total Income
            input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
            
            # Ratio d'endettement
            input_df['LoanAmountToIncome'] = input_df['LoanAmount'] / (input_df['TotalIncome'] + 1)
            
            # EMI (mensualité)
            input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
            
            # Ratio EMI / Revenu
            input_df['EMIToIncome'] = input_df['EMI'] / (input_df['TotalIncome'] + 1)
            
            # Transformations logarithmiques
            input_df['Log_LoanAmount'] = np.log(input_df['LoanAmount'] + 1)
            input_df['Log_TotalIncome'] = np.log(input_df['TotalIncome'] + 1)
            
            # Indicateur de co-demandeur
            input_df['Has_Coapplicant'] = (input_df['CoapplicantIncome'] > 0).astype(int)
            
            # -----------------------------------------------------------------
            # ÉTAPE 3 : VÉRIFIER L'ORDRE DES COLONNES
            # -----------------------------------------------------------------
            
            # TODO : S'assurer que les colonnes sont dans le bon ordre
            # (même ordre que lors de l'entraînement)
            expected_order = ['Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'TotalIncome', 'LoanAmountToIncome', 'EMI', 'EMIToIncome', 'Log_LoanAmount', 'Log_TotalIncome', 'Has_Coapplicant', 'Area_Semiurban', 'Area_Urban', 'Gender_Male', 'Married_Yes', 'SelfEmployed_Yes']
            # Vérifier si le modèle a feature_names_in_ (scikit-learn >= 1.0)
            if hasattr(model, 'feature_names_in_'):
                # Utiliser l'ordre exact du modèle
                input_df = input_df[model.feature_names_in_]
            else:
                # Fallback : utiliser l'ordre défini manuellement
                input_df = input_df[expected_order]
        
            
            # -----------------------------------------------------------------
            # ÉTAPE 4 : NORMALISATION (si Logistic Regression)
            # -----------------------------------------------------------------
            
            if model_choice == "Logistic Regression" and scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            
            # -----------------------------------------------------------------
            # ÉTAPE 5 : PRÉDICTION
            # -----------------------------------------------------------------
            
            try:
                # Faire la prédiction
                if model_choice == "Logistic Regression" and scaler is not None:
                    prediction = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0]
                else:
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                
                # Probabilités
                proba_rejected = proba[0] * 100
                proba_approved = proba[1] * 100
                
                # =============================================================
                # AFFICHAGE DES RÉSULTATS
                # =============================================================
                
                st.subheader("📊 Résultat de la prédiction")
                
                # Afficher le résultat avec un style visuel fort
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.success("### ✅ PRÊT APPROUVÉ")
                        st.balloons()  # Animation de célébration !
                    else:
                        st.error("### ❌ PRÊT REJETÉ")
                
                st.markdown("---")
                
                # Afficher les probabilités
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Probabilité d'approbation",
                        value=f"{proba_approved:.1f}%",
                        help="Confiance du modèle dans l'approbation"
                    )
                    st.progress(proba_approved / 100)
                
                with col2:
                    st.metric(
                        label="Probabilité de rejet",
                        value=f"{proba_rejected:.1f}%",
                        help="Confiance du modèle dans le rejet"
                    )
                    st.progress(proba_rejected / 100)
                
                # Niveau de confiance
                confidence = max(proba_approved, proba_rejected)
                if confidence > 90:
                    st.info("💪 **Niveau de confiance** : Très élevé")
                elif confidence > 70:
                    st.info("👍 **Niveau de confiance** : Élevé")
                elif confidence > 60:
                    st.warning("🤔 **Niveau de confiance** : Modéré")
                else:
                    st.warning("⚠️ **Niveau de confiance** : Faible - Décision incertaine")
                
                st.markdown("---")
                
                # =============================================================
                # EXPLICATION DE LA DÉCISION
                # =============================================================
                
                st.subheader("🔍 Explication de la décision")
                st.write("Voici les facteurs qui ont le plus influencé cette prédiction :")
                
                if model_choice == "Logistic Regression":
                    # Pour la régression logistique, calculer l'impact de chaque feature
                    coefficients = model.coef_[0]
                    
                    # Calculer l'impact (valeur normalisée * coefficient)
                    if scaler is not None:
                        impacts = input_scaled[0] * coefficients
                    else:
                        impacts = input_df.values[0] * coefficients
                    
                    # Créer un DataFrame avec les impacts
                    impact_df = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Impact': impacts
                    }).sort_values('Impact', key=abs, ascending=False).head(5)
                    
                    # Renommer les features pour plus de clarté
                    feature_names_mapping = {
                        'Credit_History': 'Historique de crédit',
                        'Log_TotalIncome': 'Revenu total (log)',
                        'LoanAmountToIncome': 'Ratio montant/revenu',
                        'EMIToIncome': 'Ratio mensualité/revenu',
                        'Education': 'Niveau d\'éducation',
                        'Married_Yes': 'Statut marital',
                        'Has_Coapplicant': 'Présence co-demandeur'
                    }
                    
                    impact_df['Feature_Label'] = impact_df['Feature'].map(
                        lambda x: feature_names_mapping.get(x, x)
                    )
                    
                    # Créer le graphique
                    fig = px.bar(
                        impact_df,
                        x='Impact',
                        y='Feature_Label',
                        orientation='h',
                        title='Top 5 des facteurs influents',
                        labels={'Feature_Label': 'Variable', 'Impact': 'Contribution'},
                        color='Impact',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0
                    )
                    
                    fig.update_layout(
                        xaxis_title='Contribution à la décision',
                        yaxis_title='',
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interprétation textuelle
                    st.write("**💡 Interprétation** :")
                    top_factor = impact_df.iloc[0]
                    if top_factor['Impact'] > 0:
                        st.success(f"✅ **{top_factor['Feature_Label']}** a le plus contribué à l'approbation")
                    else:
                        st.error(f"❌ **{top_factor['Feature_Label']}** a le plus contribué au rejet")
                
                else:
                    # Pour Random Forest, afficher les feature importances globales
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    fig = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Variables les plus importantes (modèle global)',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("ℹ️ Note : Pour Random Forest, on affiche l'importance globale des variables (pas spécifique à cette prédiction)")
                
                st.markdown("---")
                
                # =============================================================
                # DÉTAILS DE LA DEMANDE
                # =============================================================
                
                with st.expander("📋 Voir les détails de la demande"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Informations financières :**")
                        st.write(f"- Revenu demandeur : {applicant_income:,} €")
                        st.write(f"- Revenu co-demandeur : {coapplicant_income:,} €")
                        st.write(f"- **Revenu total : {applicant_income + coapplicant_income:,} €**")
                        st.write(f"- Montant prêt : {loan_amount:,} €")
                        st.write(f"- Durée : {loan_term} mois ({loan_term/12:.1f} ans)")
                        st.write(f"- **Mensualité estimée : {loan_amount/loan_term:,.0f} €**")
                        st.write(f"- **Ratio d'endettement : {(loan_amount/loan_term)/(applicant_income + coapplicant_income)*100:.1f}%**")
                    
                    with col2:
                        st.write("**Informations personnelles :**")
                        st.write(f"- Historique crédit : {'✅ Bon' if credit_history == 1 else '❌ Mauvais'}")
                        st.write(f"- Éducation : {'🎓 Graduate' if education == 1 else '📚 Not Graduate'}")
                        st.write(f"- Statut marital : {'💑 Marié(e)' if married == 1 else '🧍 Célibataire'}")
                        st.write(f"- Personnes à charge : {dependents}")
                        st.write(f"- Indépendant : {'✅ Oui' if self_employed == 1 else '❌ Non'}")
                        st.write(f"- Zone bien : {property_area}")
                
                # =============================================================
                # VALIDATIONS ET WARNINGS
                # =============================================================
                
                # Vérifications de cohérence
                warnings = []
                
                if applicant_income < 1000:
                    warnings.append("⚠️ Le revenu du demandeur est très faible")
                
                if loan_amount / loan_term > applicant_income + coapplicant_income:
                    warnings.append("⚠️ La mensualité dépasse le revenu total mensuel")
                
                if (loan_amount / loan_term) / (applicant_income + coapplicant_income) > 0.4:
                    warnings.append("⚠️ Le taux d'endettement dépasse 40% (seuil bancaire standard)")
                
                if loan_amount > (applicant_income + coapplicant_income) * 120:
                    warnings.append("⚠️ Le montant du prêt est très élevé par rapport au revenu")
                
                if warnings:
                    st.warning("**⚠️ Points d'attention détectés :**")
                    for warning in warnings:
                        st.write(warning)
                
            except Exception as e:
                st.error(f"❌ **Erreur lors de la prédiction**")
                st.error(f"Message d'erreur : {str(e)}")
                
                # Afficher des informations de debug
                with st.expander("🐛 Informations de débogage"):
                    st.write("**Colonnes du DataFrame d'input :**")
                    st.write(input_df.columns.tolist())
                    st.write("**Shape :**", input_df.shape)
                    st.write("**Colonnes attendues par le modèle :**")
                    try:
                        st.write(model.feature_names_in_.tolist())
                    except:
                        st.write("Non disponible")

with tab3:
    st.header("Performance du modèle")
    st.write("Section à compléter")

## reponsive design & accessibility

# charger les métriques pré-calculées
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# prédire
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calcul des métriques
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)