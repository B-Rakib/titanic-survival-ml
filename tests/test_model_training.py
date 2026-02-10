"""
Tests unitaires pour le module model_training.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import (
    create_model,
    train_model,
    save_model,
    load_model,
    predict
)


class TestCreateModel:
    """Tests pour create_model."""

    def test_returns_logistic_regression(self):
        """Vérifie que create_model retourne LogisticRegression."""
        model = create_model()
        assert isinstance(model, LogisticRegression)

    def test_model_has_correct_params(self):
        """Vérifie que le modèle a les bons paramètres."""
        model = create_model(random_state=42)
        assert model.random_state == 42
        assert model.max_iter == 1000


class TestTrainModel:
    """Tests pour train_model."""

    def test_model_is_fitted(self):
        """Vérifie que le modèle est entraîné."""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        model = create_model()
        model = train_model(model, X, y)

        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')


class TestSaveLoadModel:
    """Tests pour save et load model."""

    def test_save_and_load_model(self, tmp_path):
        """Vérifie qu'on peut sauvegarder et charger un modèle."""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        model = create_model()
        model = train_model(model, X, y)

        model_path = tmp_path / "test_model.pkl"
        save_model(model, str(model_path))

        loaded_model = load_model(str(model_path))

        assert isinstance(loaded_model, LogisticRegression)
        assert hasattr(loaded_model, 'coef_')


class TestPredict:
    """Tests pour predict."""

    def test_predict_returns_array(self):
        """Vérifie que predict retourne un array."""
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.rand(20, 5))

        model = create_model()
        model = train_model(model, X_train, y_train)

        predictions = predict(model, X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 20

    def test_predictions_are_binary(self):
        """Vérifie que les prédictions sont 0 ou 1."""
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.rand(20, 5))

        model = create_model()
        model = train_model(model, X_train, y_train)

        predictions = predict(model, X_test)

        assert set(predictions).issubset({0, 1})