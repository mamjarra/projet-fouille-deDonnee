import streamlit as st
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# ------------------------
# Charger Word2Vec & LabelEncoder
# ------------------------
w2v_model = Word2Vec.load("src/vectorizer.model")
label_encoder = joblib.load("src/label_encoder.pkl")

# ------------------------
# Fonctions
# ------------------------
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text.split()

def vectorize(text):
    tokens = preprocess(text)
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# 🔴 Liste des labels considérés comme anomalies
anomaly_labels = [
    "authentication-failed",
    "http-request-failure",
    "ids-alert",
    "file-action-failure",
    "process-error",
    "connection-failed"
]

# ------------------------
# Interface utilisateur
# ------------------------
st.set_page_config(page_title="SIEM Log Analyzer", page_icon="🛡️", layout="centered")
st.title("🛡️ Analyse de Logs SIEM")
st.subheader("Détection intelligente des anomalies 🔍")
st.markdown("---")

# Mode d'entrée : texte ou fichier
input_mode = st.radio("📥 Choisissez le mode d'entrée :", ("📝 Texte manuel", "📂 Fichier .txt"))

log_input = ""
logs_list = []

if input_mode == "📝 Texte manuel":
    log_input = st.text_area("Entrez un log à analyser :", height=150, placeholder="Ex : Failed password for root from 192.168.1.1 port 22 ssh2")
else:
    uploaded_file = st.file_uploader("Téléversez un fichier .txt avec un log par ligne", type=["txt"])
    if uploaded_file is not None:
        logs_list = uploaded_file.read().decode("utf-8").splitlines()

model_paths = {
    "🔴 Surappris": "src/model.pkl",
    "🟡 Régularisé": "src/model_regul.pkl",
    "🟢 Validation croisée": "src/model_cv.pkl"
}
model_choice = st.selectbox("Choisissez le modèle d'IA :", list(model_paths.keys()))

if st.button("🔍 Lancer la prédiction"):
    try:
        clf = joblib.load(model_paths[model_choice])

        # Mode texte simple
        if input_mode == "📝 Texte manuel":
            if not log_input.strip():
                st.warning("⚠️ Veuillez entrer un log.")
            else:
                tokens = preprocess(log_input)
                vec = vectorize(log_input).reshape(1, -1)
                pred_class = clf.predict(vec)[0]
                label = label_encoder.inverse_transform([pred_class])[0]

                st.subheader("🔎 Résultat de l'analyse")
                if label in anomaly_labels:
                    st.error(f"🚨 Anomalie détectée ! Label : `{label}`")
                else:
                    st.success(f"✅ Log normal. Label : `{label}`")

                st.markdown("### 📊 Analyse du log")
                col1, col2 = st.columns(2)
                col1.metric("Longueur du log (caractères)", len(log_input))
                col2.metric("Nombre de mots (tokens)", len(tokens))

                with st.expander("🧬 Tokens extraits"):
                    st.write(tokens)

                with st.expander("📈 Vecteur Word2Vec"):
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(vec.flatten(), marker='o')
                    ax.set_title("Vecteur moyen Word2Vec du log")
                    st.pyplot(fig)

        # Mode fichier .txt
        else:
            if not logs_list:
                st.warning("⚠️ Aucun fichier ou fichier vide.")
            else:
                st.success(f"✅ {len(logs_list)} logs chargés")
                for idx, log in enumerate(logs_list, 1):
                    tokens = preprocess(log)
                    vec = vectorize(log).reshape(1, -1)
                    pred_class = clf.predict(vec)[0]
                    label = label_encoder.inverse_transform([pred_class])[0]

                    st.markdown(f"### 🔍 Log #{idx}")
                    if label in anomaly_labels:
                        st.error(f"🚨 Anomalie détectée ! Label : `{label}`")
                    else:
                        st.success(f"✅ Log normal. Label : `{label}`")
                    st.write(log)
                    st.markdown("---")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
