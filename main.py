"""
Script principal du projet Titanic Survival Prediction.

Ce script orchestre l'exécution complète du pipeline:
1. Prétraitement des données
2. Entraînement du modèle
3. Évaluation et génération de la soumission
"""

import sys
import os

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import preprocess_pipeline
from model_training import training_pipeline
from model_evaluation import evaluation_pipeline


def main():
    """
    Fonction principale qui exécute le pipeline complet.
    """
    print("\n" + "🚢 "*20)
    print("TITANIC SURVIVAL PREDICTION - PIPELINE COMPLET")
    print("🚢 "*20 + "\n")
    
    # Étape 1: Prétraitement
    print("\n📊 ÉTAPE 1/3: Prétraitement des données\n")
    preprocess_pipeline(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        output_train="data/processed/train_processed.csv",
        output_test="data/processed/test_processed.csv"
    )
    
    # Étape 2: Entraînement
    print("\n🤖 ÉTAPE 2/3: Entraînement du modèle\n")
    training_pipeline(
        train_data_path="data/processed/train_processed.csv",
        model_output_path="models/titanic_model.pkl"
    )
    
    # Étape 3: Évaluation et Soumission
    print("\n📈 ÉTAPE 3/3: Évaluation et génération soumission\n")
    evaluation_pipeline(
        model_path="models/titanic_model.pkl",
        test_data_path="data/processed/test_processed.csv",
        submission_path="data/processed/submission.csv"
    )
    
    print("\n" + "✅ "*20)
    print("PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("✅ "*20 + "\n")
    print("📁 Fichiers générés:")
    print("  - data/processed/train_processed.csv")
    print("  - data/processed/test_processed.csv")
    print("  - models/titanic_model.pkl")
    print("  - data/processed/submission.csv")
    print("\n")


if __name__ == "__main__":
    main()
