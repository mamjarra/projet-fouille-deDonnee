import streamlit as st
import joblib

# Chargement du modèle et du vectoriseur
model = joblib.load('models/vote_model.pkl')
vect = joblib.load('models/tfidf_vect.pkl')

st.title("🧠 Prédiction de Classe de Texte")
st.write("Entrez un texte à classer selon le modèle IA.")

text = st.text_area("Texte à analyser")

if st.button("Prédire"):
    if text.strip():
        X = vect.transform([text])
        pred = model.predict(X)[0]
        st.success(f"Classe prédite : {pred}")
    else:
        st.warning("Veuillez entrer un texte valide.")