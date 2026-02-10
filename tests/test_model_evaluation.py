"""
Tests unitaires pour le module model_evaluation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from model_evaluation import calculate_metrics, create_submission


class TestCalculateMetrics:
    """Tests pour calculate_metrics."""

    def test_metrics_dict_returned(self):
        """Vérifie qu'un dictionnaire de métriques est retourné."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_metrics_values_valid(self):
        """Vérifie que les métriques sont entre 0 et 1."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)

        for value in metrics.values():
            assert 0 <= value <= 1


class TestCreateSubmission:
    """Tests pour create_submission."""

    def test_submission_file_created(self, tmp_path):
        """Vérifie que le fichier de soumission est créé."""
        passenger_ids = pd.Series([892, 893, 894, 895])
        predictions = np.array([0, 1, 0, 1])

        output_path = tmp_path / "test_submission.csv"
        create_submission(passenger_ids, predictions, str(output_path))

        assert output_path.exists()

    def test_submission_format_correct(self, tmp_path):
        """Vérifie le format du fichier de soumission."""
        passenger_ids = pd.Series([892, 893, 894, 895])
        predictions = np.array([0, 1, 0, 1])

        output_path = tmp_path / "test_submission.csv"
        create_submission(passenger_ids, predictions, str(output_path))

        df = pd.read_csv(output_path)

        assert "PassengerId" in df.columns
        assert "Survived" in df.columns
        assert len(df) == 4
        assert df["Survived"].isin([0, 1]).all()
