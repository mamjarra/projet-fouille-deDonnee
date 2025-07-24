import pandas as pd
import numpy as np
import re
import os
import joblib
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

# === Étape 1 : Charger le fichier brut ===
log_file = "data/output_0.1.log"

# Lire le fichier avec les colonnes: category,log,tokens (on ignore tokens)
df = pd.read_csv(log_file, sep=",", usecols=[0, 1], names=["Label", "Content"], header=0, on_bad_lines="skip")

# Nettoyer les valeurs manquantes
df.dropna(inplace=True)

# === Étape 2 : Prétraitement du texte ===
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text.split()

df["tokens"] = df["Content"].apply(preprocess)

# === Étape 3 : Entraînement du modèle Word2Vec ===
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4)
os.makedirs("src", exist_ok=True)
w2v_model.save("src/vectorizer.model")

# === Étape 4 : Vectorisation moyenne des tokens ===
def vectorize(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.array([vectorize(toks) for toks in df["tokens"]])

# === Étape 5 : Encodage des étiquettes ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Label"])

# === Étape 6 : Sauvegarde ===
np.save("src/X.npy", X)
np.save("src/y.npy", y)
joblib.dump(label_encoder, "src/label_encoder.pkl")  # ⬅️ Sauvegarde du label_encoder
df[["Label", "Content", "tokens"]].to_csv("data/log_structured_labeled.csv", index=False)

print("✅ Dataset vectorisé et sauvegardé avec succès !")
print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
