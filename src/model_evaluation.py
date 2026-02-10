"""
Module d'évaluation du modèle Titanic.

Ce module contient les fonctions pour évaluer le modèle et
générer les fichiers de soumission pour Kaggle.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict


def load_model(filepath: str) -> Any:
    """
    Charge le modèle sauvegardé.

    Args:
        filepath: Chemin vers le modèle

    Returns:
        Modèle chargé
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"✓ Modèle chargé: {filepath}")
    return model


def load_test_data(filepath: str) -> tuple:
    """
    Charge les données de test.

    Args:
        filepath: Chemin vers test_processed.csv

    Returns:
        Tuple (passenger_ids, X_test)
    """
    df = pd.read_csv(filepath)
    passenger_ids = df["PassengerId"]
    X_test = df.drop("PassengerId", axis=1)

    print(f"✓ Données de test chargées: {len(X_test)} exemples")
    return passenger_ids, X_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calcule les métriques d'évaluation.

    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        y_proba: Probabilités (optionnel)

    Returns:
        Dictionnaire des métriques
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Affiche les métriques de manière formatée.

    Args:
        metrics: Dictionnaire des métriques
    """
    print("\n" + "=" * 50)
    print("MÉTRIQUES D'ÉVALUATION")
    print("=" * 50 + "\n")

    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():.<20} {value:.4f}")

    print("\n" + "=" * 50 + "\n")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = None) -> None:
    """
    Crée et affiche la matrice de confusion.

    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        output_path: Chemin pour sauvegarder (optionnel)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Décédé", "Survécu"], yticklabels=["Décédé", "Survécu"]
    )
    plt.title("Matrice de Confusion")
    plt.ylabel("Vraie Valeur")
    plt.xlabel("Prédiction")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Matrice de confusion sauvegardée: {output_path}")

    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str = None) -> None:
    """
    Crée et affiche la courbe ROC.

    Args:
        y_true: Vraies valeurs
        y_proba: Probabilités de prédiction
        output_path: Chemin pour sauvegarder (optionnel)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Courbe ROC sauvegardée: {output_path}")

    plt.close()


def create_submission(passenger_ids: pd.Series, predictions: np.ndarray, output_path: str) -> None:
    """
    Crée le fichier de soumission Kaggle.

    Args:
        passenger_ids: IDs des passagers
        predictions: Prédictions du modèle
        output_path: Chemin de sauvegarde
    """
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})

    submission.to_csv(output_path, index=False)

    print(f"\n✓ Fichier de soumission créé: {output_path}")
    print(f"✓ Prédictions: {predictions.sum()} survivants sur {len(predictions)}")


def evaluation_pipeline(
    model_path: str, test_data_path: str, submission_path: str, plots_dir: str = "docs/plots"
) -> None:
    """
    Pipeline complet d'évaluation.

    Args:
        model_path: Chemin vers le modèle
        test_data_path: Chemin vers données de test
        submission_path: Chemin fichier soumission
        plots_dir: Dossier pour les graphiques
    """
    print("\n" + "=" * 50)
    print("PIPELINE D'ÉVALUATION")
    print("=" * 50 + "\n")

    # Charger le modèle
    model = load_model(model_path)

    # Charger les données de test
    passenger_ids, X_test = load_test_data(test_data_path)

    # Faire des prédictions
    predictions = model.predict(X_test)

    print(f"✓ Prédictions générées: {len(predictions)} exemples")

    # Créer le fichier de soumission
    create_submission(passenger_ids, predictions, submission_path)

    print("\n" + "=" * 50)
    print("ÉVALUATION TERMINÉE")
    print("=" * 50 + "\n")


def evaluate_on_validation(model_path: str, validation_data_path: str, plots_dir: str = "docs/plots") -> None:
    """
    Évalue le modèle sur un ensemble de validation.

    Args:
        model_path: Chemin vers le modèle
        validation_data_path: Chemin données validation
        plots_dir: Dossier pour les graphiques
    """
    import os

    os.makedirs(plots_dir, exist_ok=True)

    # Charger le modèle
    model = load_model(model_path)

    # Charger les données
    df = pd.read_csv(validation_data_path)
    X_val = df.drop(["PassengerId", "Survived"], axis=1)
    y_val = df["Survived"]

    # Prédictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Calculer métriques
    metrics = calculate_metrics(y_val, y_pred, y_proba)
    print_metrics(metrics)

    # Générer graphiques
    plot_confusion_matrix(y_val, y_pred, f"{plots_dir}/confusion_matrix.png")
    plot_roc_curve(y_val, y_proba, f"{plots_dir}/roc_curve.png")


if __name__ == "__main__":
    # Exécution du pipeline d'évaluation
    evaluation_pipeline(
        model_path="models/titanic_model.pkl",
        test_data_path="data/processed/test_processed.csv",
        submission_path="data/processed/submission.csv",
    )
