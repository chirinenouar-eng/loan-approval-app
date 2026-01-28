import streamlit as st

# ⚠️ Toujours mettre la config en tout début de fichier
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🚀",
    layout="wide"
)

# contenu
st.write("Bienvenue dans l'application de prédiction d'approbation de prêt.")

# mot de passe
password = st.text_input("Entrez le mot de passe pour accéder à l'application :", type="password")

if password == st.secrets["password"]:
    st.success("Accès accordé ! Vous pouvez maintenant utiliser l'application.")

    # titre
    st.title("Loan Approval Prediction App 🚀")

else:
    st.error("Mot de passe incorrect. Veuillez réessayer.")
    st.stop()