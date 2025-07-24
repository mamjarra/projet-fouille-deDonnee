import streamlit as st
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# ------------------------
# 🎨 Mise en page
# ------------------------
st.set_page_config(page_title="SIEM Log Analyzer", page_icon="🛡️", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTextArea textarea { font-size: 16px; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------
# 📦 Charger Word2Vec & LabelEncoder
# ------------------------
w2v_model = Word2Vec.load("src/vectorizer.model")
label_encoder = joblib.load("src/label_encoder.pkl")  # Assure-toi que ce fichier est bien généré dans prepare_dataset.py

# ------------------------
# ⚙️ Fonctions
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
st.title("🛡️ Analyse de Logs SIEM")
st.subheader("Détection intelligente des anomalies 🔍")
st.markdown("---")

log_input = st.text_area("📝 Entrez un log à analyser :", height=150, placeholder="Ex : Failed password for root from 192.168.1.1 port 22 ssh2")

model_paths = {
    "🔴 Surappris": "src/model.pkl",
    "🟡 Régularisé": "src/model_regul.pkl",
    "🟢 Validation croisée": "src/model_cv.pkl"
}

model_choice = st.selectbox("Choisissez le modèle d'IA :", list(model_paths.keys()))

if st.button("🔍 Lancer la prédiction"):
    if not log_input.strip():
        st.warning("⚠️ Veuillez entrer un log.")
    else:
        try:
            with st.spinner("⏳ Analyse en cours..."):
                tokens = preprocess(log_input)
                vec = vectorize(log_input).reshape(1, -1)
                clf = joblib.load(model_paths[model_choice])
                pred_class = clf.predict(vec)[0]
                label = label_encoder.inverse_transform([pred_class])[0]


            # Affichage résultat
            st.subheader("🔎 Résultat de l'analyse")
            if label in anomaly_labels:
                st.error(f"🚨 Anomalie détectée ! Label : `{label}`")
            else:
                st.success(f"✅ Log normal. Label : `{label}`")
                st.balloons()

            # 🔍 Analyse détaillée
            st.markdown("### 📊 Analyse du log")
            col1, col2 = st.columns(2)
            col1.metric("Longueur du log (caractères)", len(log_input))
            col2.metric("Nombre de mots (tokens)", len(tokens))

            with st.expander("🧬 Afficher les tokens extraits"):
                st.write(tokens)

            with st.expander("📈 Afficher la courbe du vecteur (100 dimensions)"):
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(vec.flatten(), marker='o')
                ax.set_title("Vecteur moyen Word2Vec du log")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
