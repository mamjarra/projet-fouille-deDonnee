import pandas as pd
import numpy as np
import re
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# === Étape 1 : Charger les données ===
df = pd.read_csv("data/log_structured_labeled.csv")
df = df[["Content", "Label"]].dropna()

# === Étape 2 : Prétraitement
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    return text.split()

df["tokens"] = df["Content"].apply(preprocess)

# === Étape 3 : Charger le modèle Word2Vec préentraîné
w2v_model = Word2Vec.load("src/vectorizer.model")

# === Étape 4 : Vectorisation
def vectorize(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.vstack(df["tokens"].apply(vectorize))
y = df["Label"].values

# === Étape 5 : Modèle SVM avec régularisation modérée
clf = SVC(C=0.5, kernel='linear')

# === Étape 6 : Validation croisée
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print("✅ Accuracy (CV folds):", scores)
print("✅ Moyenne:", np.mean(scores))
print("✅ Écart-type:", np.std(scores))

# === Étape 7 : Entraînement final sur tout le dataset
clf.fit(X, y)
joblib.dump(clf, "src/model_cv.pkl")
print("✅ Modèle cross-validé sauvegardé sous src/model_cv.pkl")
