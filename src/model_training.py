"""
Module d'entraînement du modèle de prédiction Titanic.

Ce module contient les fonctions pour entraîner, sauvegarder et
charger le modèle de Machine Learning.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Any


def load_processed_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Charge les données prétraitées et sépare features/target.

    Args:
        filepath: Chemin vers le fichier CSV traité

    Returns:
        Tuple (X, y) où X sont les features et y la target
    """
    df = pd.read_csv(filepath)

    X = df.drop(['PassengerId', 'Survived'], axis=1)
    y = df['Survived']

    print("Données chargées: {} exemples, {} features".format(X.shape[0], X.shape[1]))
    return X, y


def create_model(random_state: int = 42) -> LogisticRegression:
    """
    Crée un modèle de Régression Logistique.

    Args:
        random_state: Seed pour reproductibilité

    Returns:
        Instance du modèle LogisticRegression
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='liblinear'
    )

    print("Modèle créé: Logistic Regression")
    return model


def train_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """
    Entraîne le modèle sur les données.

    Args:
        model: Modèle à entraîner
        X: Features d'entraînement
        y: Target d'entraînement

    Returns:
        Modèle entraîné
    """
    print("\n" + "=" * 50)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("=" * 50 + "\n")

    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("Modèle entraîné")
    print("Cross-validation scores: {}".format(cv_scores))
    print("Accuracy moyenne: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std()))

    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Évalue le modèle sur les données.

    Args:
        model: Modèle entraîné
        X: Features de test
        y: Target de test

    Returns:
        Score d'accuracy
    """
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print("\n" + "=" * 50)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 50 + "\n")
    print("Accuracy: {:.4f}".format(accuracy))
    print("\nRapport de classification:")
    print(classification_report(y, predictions,
                                target_names=['Décédé', 'Survécu']))

    return accuracy


def save_model(model: Any, filepath: str) -> None:
    """
    Sauvegarde le modèle entraîné.

    Args:
        model: Modèle à sauvegarder
        filepath: Chemin de sauvegarde
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print("\nModèle sauvegardé: {}".format(filepath))


def load_model(filepath: str) -> Any:
    """
    Charge un modèle sauvegardé.

    Args:
        filepath: Chemin vers le modèle

    Returns:
        Modèle chargé
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print("Modèle chargé: {}".format(filepath))
    return model


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Fait des prédictions avec le modèle.

    Args:
        model: Modèle entraîné
        X: Features pour prédiction

    Returns:
        Array des prédictions
    """
    predictions = model.predict(X)
    print("{} prédictions générées".format(len(predictions)))

    return predictions


def training_pipeline(train_data_path: str,
                      model_output_path: str,
                      test_size: float = 0.2) -> None:
    """
    Pipeline complet d'entraînement.

    Args:
        train_data_path: Chemin données d'entraînement
        model_output_path: Chemin sauvegarde modèle
        test_size: Proportion pour validation
    """
    print("\n" + "=" * 50)
    print("PIPELINE D'ENTRAÎNEMENT")
    print("=" * 50 + "\n")

    X, y = load_processed_data(train_data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print("Split: {} train, {} validation".format(len(X_train), len(X_val)))

    model = create_model()
    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_val, y_val)

    save_model(model, model_output_path)

    print("\n" + "=" * 50)
    print("PIPELINE TERMINÉ")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    training_pipeline(
        train_data_path="data/processed/train_processed.csv",
        model_output_path="models/titanic_model.pkl"
    )