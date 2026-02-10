"""
Tests unitaires pour le module data_preprocessing.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    load_data,
    handle_missing_values,
    create_features,
    encode_categorical,
    select_features
)


class TestLoadData:
    """Tests pour la fonction load_data."""

    def test_load_data_returns_dataframes(self):
        """Vérifie que load_data retourne deux DataFrames."""
        train_df, test_df = load_data(
            "data/raw/train.csv",
            "data/raw/test.csv"
        )
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_load_data_correct_size(self):
        """Vérifie que les données ont la bonne taille."""
        train_df, test_df = load_data(
            "data/raw/train.csv",
            "data/raw/test.csv"
        )
        assert len(train_df) == 891
        assert len(test_df) == 418


class TestHandleMissingValues:
    """Tests pour handle_missing_values."""

    def test_has_cabin_feature_created(self):
        """Vérifie que Has_Cabin est créé."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = handle_missing_values(train_df)
        assert 'Has_Cabin' in df.columns
        assert df['Has_Cabin'].dtype == int


class TestCreateFeatures:
    """Tests pour create_features."""

    def test_family_size_created(self):
        """Vérifie que FamilySize est créé."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = create_features(train_df)
        assert 'FamilySize' in df.columns

    def test_is_alone_created(self):
        """Vérifie que IsAlone est créé."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = create_features(train_df)
        assert 'IsAlone' in df.columns
        assert df['IsAlone'].isin([0, 1]).all()

    def test_title_extracted(self):
        """Vérifie que Title est extrait."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = create_features(train_df)
        assert 'Title' in df.columns


class TestEncodeCategorical:
    """Tests pour encode_categorical."""

    def test_sex_becomes_numeric(self):
        """Vérifie que Sex devient numérique."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = handle_missing_values(train_df)
        df = create_features(train_df)
        df = encode_categorical(df)
        assert pd.api.types.is_numeric_dtype(df['Sex'])
        assert set(df['Sex'].unique()).issubset({0, 1})


class TestSelectFeatures:
    """Tests pour select_features."""

    def test_survived_in_train(self):
        """Vérifie que Survived est présent dans train."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = handle_missing_values(train_df)
        df = create_features(df)
        df = encode_categorical(df)
        df_selected = select_features(df, is_train=True)
        assert 'Survived' in df_selected.columns

    def test_passenger_id_present(self):
        """Vérifie que PassengerId est présent."""
        train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")
        df = handle_missing_values(train_df)
        df = create_features(df)
        df = encode_categorical(df)
        df_selected = select_features(df, is_train=True)
        assert 'PassengerId' in df_selected.columns