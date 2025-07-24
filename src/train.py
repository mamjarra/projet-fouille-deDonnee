import pandas as pd
import numpy as np
import re
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# === Étape 1 : Charger les données ===
df = pd.read_csv("data/log_structured_labeled.csv")
df = df[["Content", "Label"]].dropna()

# === Étape 2 : Prétraitement identique ===
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text.split()

df["tokens"] = df["Content"].apply(preprocess)

# === Étape 3 : Charger le modèle Word2Vec préentraîné ===
w2v_model = Word2Vec.load("src/vectorizer.model")

# === Étape 4 : Vectorisation moyenne ===
def vectorize(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.vstack(df["tokens"].apply(vectorize))
y = df["Label"].values

# === Étape 5 : Simulation du surapprentissage (jeu d'entraînement réduit) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42)

# SVM avec peu de régularisation (C élevé)
clf = SVC(C=1000, kernel='linear')
clf.fit(X_train, y_train)

# === Étape 6 : Évaluation ===
y_pred = clf.predict(X_test)
print("✅ Accuracy (test set):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Étape 7 : Sauvegarde ===
joblib.dump(clf, "src/model.pkl")
print("✅ Modèle surentraîné sauvegardé sous src/model.pkl")
