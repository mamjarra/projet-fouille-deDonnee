import pandas as pd
import numpy as np
import re
import joblib
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, accuracy_score

# === Étape 1 : Charger les données ===
df = pd.read_csv("data/log_structured_labeled.csv")
df = df[["Content", "Label"]].dropna()

# === Étape 2 : Prétraitement
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text.split()

df["tokens"] = df["Content"].apply(preprocess)

# === Étape 3 : Charger Word2Vec existant
w2v_model = Word2Vec.load("src/vectorizer.model")

# === Étape 4 : Vectorisation
def vectorize(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.vstack(df["tokens"].apply(vectorize))
y = df["Label"].values

# === Étape 5 : Chargement des modèles
models = {
    "Surappris": "src/model.pkl",
    "Régularisé": "src/model_regul.pkl",
    "Validation croisée": "src/model_cv.pkl"
}

# === Étape 6 : Évaluation des modèles
print("📊 Comparaison des modèles :\n")
for name, path in models.items():
    clf = joblib.load(path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"🔹 {name} - Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    print("-" * 60)
