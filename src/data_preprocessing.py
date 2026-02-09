"""
Module de prétraitement des données Titanic.

Ce module contient les fonctions pour charger, nettoyer et préparer
les données du Titanic pour l'entraînement du modèle.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données d'entraînement et de test.
    
    Args:
        train_path: Chemin vers le fichier train.csv
        test_path: Chemin vers le fichier test.csv
        
    Returns:
        Tuple contenant (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Données chargées: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le dataset.
    
    Args:
        df: DataFrame à traiter
        
    Returns:
        DataFrame avec valeurs manquantes traitées
    """
    df = df.copy()
    
    # Age: remplacer par la médiane
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: remplacer par le mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fare: remplacer par la médiane
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Cabin: créer une feature binaire
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df = df.drop('Cabin', axis=1)
    
    print(f"✓ Valeurs manquantes traitées")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des données existantes.
    
    Args:
        df: DataFrame source
        
    Returns:
        DataFrame avec nouvelles features
    """
    df = df.copy()
    
    # FamilySize: SibSp + Parch + 1
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone: 1 si seul, 0 sinon
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title: extraire de Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Simplifier les titres rares
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                        'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                        'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    print(f"✓ Features créées: FamilySize, IsAlone, Title")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles.
    
    Args:
        df: DataFrame à encoder
        
    Returns:
        DataFrame avec variables encodées
    """
    df = df.copy()
    
    # Sex: Male=1, Female=0
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # Embarked: C=0, Q=1, S=2
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Title: One-hot encoding
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)
    
    print(f"✓ Variables catégorielles encodées")
    return df


def select_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Sélectionne les features finales pour le modèle.
    
    Args:
        df: DataFrame source
        is_train: True si dataset d'entraînement (contient 'Survived')
        
    Returns:
        DataFrame avec features sélectionnées
    """
    # Features à garder
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Has_Cabin']
    
    # Ajouter les colonnes Title_*
    title_cols = [col for col in df.columns if col.startswith('Title_')]
    features.extend(title_cols)
    
    if is_train and 'Survived' in df.columns:
        features.insert(0, 'Survived')
    
    # Garder PassengerId pour la soumission
    if 'PassengerId' in df.columns:
        features.insert(0, 'PassengerId')
    
    df_selected = df[features].copy()
    print(f"✓ {len(features)} features sélectionnées")
    
    return df_selected


def preprocess_pipeline(train_path: str, test_path: str, 
                       output_train: str, output_test: str) -> None:
    """
    Pipeline complet de prétraitement.
    
    Args:
        train_path: Chemin train.csv
        test_path: Chemin test.csv
        output_train: Chemin sortie train traité
        output_test: Chemin sortie test traité
    """
    print("\n" + "="*50)
    print("PRÉTRAITEMENT DES DONNÉES")
    print("="*50 + "\n")
    
    # Charger
    train_df, test_df = load_data(train_path, test_path)
    
    # Traiter valeurs manquantes
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    
    # Créer features
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Encoder
    train_df = encode_categorical(train_df)
    test_df = encode_categorical(test_df)
    
    # Sélectionner features
    train_processed = select_features(train_df, is_train=True)
    test_processed = select_features(test_df, is_train=False)
    
    # Sauvegarder
    train_processed.to_csv(output_train, index=False)
    test_processed.to_csv(output_test, index=False)
    
    print(f"\n✓ Données sauvegardées:")
    print(f"  - Train: {output_train}")
    print(f"  - Test: {output_test}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # Exécution du pipeline
    preprocess_pipeline(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        output_train="data/processed/train_processed.csv",
        output_test="data/processed/test_processed.csv"
    )