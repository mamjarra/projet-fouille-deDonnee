
# Détection d'Anomalies dans les Logs SIEM

## Description du Projet

Ce projet vise à détecter automatiquement les anomalies dans les logs SIEM en utilisant un algorithme **SVM** avec des vecteurs de mots Word2Vec. Le pipeline inclut la simulation du **surapprentissage**, l'application de la **régularisation** et la **validation croisée**, avec un déploiement via **Streamlit**.

## Technologies Utilisées

- Python (Pandas, NumPy, Scikit-learn, matplotlib)
- Word2Vec (Gensim)
- Streamlit pour l’interface utilisateur
- Joblib pour la sauvegarde des modèles

## Exécution du Projet

1. Préparation des Données :

   python prepare_dataset.py

2. Entraînement des Modèles :

   python train.py
   python train_regul.py
   python train_cv.py

3. **Comparaison des Modèles (optionnel) :**

   python compare_models.py

4. **Lancement de l'Application Web :**

   streamlit run app.py
