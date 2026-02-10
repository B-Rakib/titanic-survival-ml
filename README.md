\# 🚢 Titanic Survival Prediction



\[!\[CI/CD Pipeline](https://github.com/B-Rakib/titanic-survival-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/B-Rakib/titanic-survival-ml/actions/workflows/ci.yml)



Projet de prédiction de survie des passagers du Titanic utilisant le Machine Learning et les bonnes pratiques d'ingénierie logicielle.



\## 📋 Objectifs du Projet



Ce projet vise à :

\- Prédire la survie des passagers du Titanic avec un modèle de \*\*Logistic Regression\*\*

\- Appliquer les \*\*bonnes pratiques d'ingénierie logicielle\*\* (modularité, tests, CI/CD)

\- Mettre en place un \*\*pipeline complet\*\* de preprocessing, training et evaluation

\- Automatiser les tests avec \*\*GitHub Actions\*\*



\## 🎯 Résultats



\- \*\*Modèle\*\* : Logistic Regression

\- \*\*Accuracy\*\* : ~80%

\- \*\*Features\*\* : Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Has\_Cabin, Title

\- \*\*Tests\*\* : 19 tests unitaires (100% de réussite)



\## 📁 Structure du Projet

```

titanic-survival-ml/

├── src/                          # Code source

│   ├── data\_preprocessing.py     # Prétraitement des données

│   ├── model\_training.py         # Entraînement du modèle

│   └── model\_evaluation.py       # Évaluation et soumission

├── tests/                        # Tests unitaires

│   ├── test\_data\_preprocessing.py

│   ├── test\_model\_training.py

│   └── test\_model\_evaluation.py

├── data/

│   ├── raw/                      # Données brutes

│   └── processed/                # Données traitées

├── models/                       # Modèles sauvegardés

├── .github/workflows/            # CI/CD

│   └── ci.yml

├── main.py                       # Script principal

├── requirements.txt              # Dépendances

├── Dockerfile                    # Containerisation

└── README.md

```



\## 🚀 Installation



\### Prérequis

\- Python 3.11+

\- pip



\### Étapes



1\. \*\*Cloner le repository\*\*

```bash

git clone https://github.com/B-Rakib/titanic-survival-ml.git

cd titanic-survival-ml

```



2\. \*\*Installer les dépendances\*\*

```bash

pip install -r requirements.txt

```



3\. \*\*Lancer le pipeline complet\*\*

```bash

python main.py

```



\## 🧪 Tests



Lancer les tests unitaires :

```bash

pytest tests/ -v

```



Résultat : \*\*19 tests passent\*\* ✅



\## 🐳 Docker



\### Build l'image

```bash

docker build -t titanic-ml .

```



\### Lancer le container

```bash

docker run titanic-ml

```



\## 📊 Pipeline de Machine Learning



Le projet suit un pipeline en 3 étapes :



\### 1. Prétraitement (`data\_preprocessing.py`)

\- Chargement des données

\- Gestion des valeurs manquantes

\- Création de features (FamilySize, IsAlone, Title)

\- Encodage des variables catégorielles



\### 2. Entraînement (`model\_training.py`)

\- Création du modèle Logistic Regression

\- Entraînement avec validation croisée

\- Sauvegarde du modèle



\### 3. Évaluation (`model\_evaluation.py`)

\- Chargement du modèle

\- Génération des prédictions

\- Création du fichier de soumission Kaggle



\## ⚙️ CI/CD



Le projet utilise \*\*GitHub Actions\*\* pour :

\- ✅ Exécuter les tests automatiquement à chaque push

\- ✅ Vérifier la qualité du code

\- ✅ Garantir que le pipeline fonctionne



\## 👥 Équipe



* BHUIYAN Rakib - Développement du projet
* Riade EL ATTAR - ReadME + Rapport + Tests



\## 📚 Technologies Utilisées



\- \*\*Python 3.11\*\*

\- \*\*scikit-learn\*\* - Machine Learning

\- \*\*pandas\*\* - Manipulation de données

\- \*\*pytest\*\* - Tests unitaires

\- \*\*GitHub Actions\*\* - CI/CD

\- \*\*Docker\*\* - Containerisation



\## 📝 Projet



Projet académique - BUT VCOD 2025-2026



\## 📧 Contact



Pour toute question : bhu.rakib05@gmail.com

