import streamlit as st
import pandas as pd

df = pd.read_csv("loan_data.csv")

# textes et titres
#st.title("test Jjk")
#st.header("powerscaling")
#st.subheader("Gojo Satoru")
#st.text("Limitless Cursed Energy")
#st.markdown("**Domain Expansion:** Infinite Void")
#st.caption("Gojo Satoru is the most powerful characters in Jujutsu Kaisen.")
#st.code("print('Gojo Satoru')", language='python')

# données
#st.dataframe(df)
#st.table(df.head())
#st.json({"character": "Gojo Satoru", "abilities": "Limitless,  Red, Blue, Hollow Purple, Domain Expansion"})
#st.metric("Power Level", "Over 9000", "+500")

#st.image("logo.png", caption="Gojo Satoru in action")
#st.audio("gojo_theme.mp3")
#st.video("gojo_fight_scene.mp4")

##  user inputs

# boutons
#if st.button("Click Me"):
#    st.write("Button clicked!")

# slider
#age = st.slider("Age", min_value=18, max_value=100, value=30)

# input numérique
#income = st.number_input("Annual Income", min_value=0, value=3000)

# selectbox (menu déroulant)
#education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

# checkbox
#agree = st.checkbox("j'accepte les termes et conditions")

# radio buttons
#married = st.radio("Marital Status", ["Married", "Single"])

# upload de fichier
#uploaded_file = st.file_uploader("Choose a csv file")

# colonnes
#col1, col2, col3 = st.columns(3)

#with col1:
#    st.header("Column 1")
#    st.write("Contenu gauche")

#with col2:
#    st.header("Column 2")
#    st.write("Contenu centre")

#with col3:
#    st.header("Column 3")
#    st.write("Contenu droite")

# sidebar
#st.sidebar.title("Options")
#st.sidebar.selectbox("Choisir un modèle", ["Logistic Regression", "Regression", "Random Forest"])
#st.sidebar.slider("Seuil de décision", 0.0, 1.0, 0.5)

# expander 
#with st.expander("Voir les détails"):
#    st.write("Here is some more information about Gojo Satoru...")
#    st.dataframe(df)

