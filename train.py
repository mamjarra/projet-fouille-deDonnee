import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement des données
df = pd.read_csv('data/dataset.csv')
X_raw, y = df['texte'], df['label']

# Prétraitement
stop = set(stopwords.words('french'))
lemm = WordNetLemmatizer()
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([lemm.lemmatize(t) for t in tokens if t.isalpha() and t not in stop])
X = X_raw.apply(preprocess)

# Vectorisation
vect = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vect.fit_transform(X)
joblib.dump(vect, 'models/tfidf_vect.pkl')

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Modèle surentraîné (overfitting)
rf_over = RandomForestClassifier(n_estimators=500, max_depth=None)
rf_over.fit(X_train, y_train)
joblib.dump(rf_over, 'models/rf_overfit.pkl')

# Méthode 1 : régularisation
params = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_leaf': [1, 5]}
gs = GridSearchCV(RandomForestClassifier(), params, cv=3)
gs.fit(X_train, y_train)
rf_reg = gs.best_estimator_
joblib.dump(rf_reg, 'models/rf_reg.pkl')

# Méthode 2 : VotingClassifier
nb = MultinomialNB()
vote = VotingClassifier(estimators=[('rf', rf_reg), ('nb', nb)], voting='soft')
vote.fit(X_train, y_train)
joblib.dump(vote, 'models/vote_model.pkl')

# Évaluation
print("Overfit Test Score:", accuracy_score(y_test, rf_over.predict(X_test)))
print("Reg Test Score:", accuracy_score(y_test, rf_reg.predict(X_test)))
print("Vote Test Score:", accuracy_score(y_test, vote.predict(X_test)))
