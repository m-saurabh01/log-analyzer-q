# utils/models.py
"""
Model wrappers for anomaly detection on log feature vectors.

Provided:
  - IsolationForestDetector (scikit-learn)
  - AutoencoderDetector (TensorFlow/Keras)

Both expose a common interface:
    preds, scores = model.fit_predict(X)

Where:
    preds  : 1 for anomaly, 0 for normal
    scores : anomaly score (higher = more anomalous)
"""

import numpy as np
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class IsolationForestDetector:
    """
    Thin wrapper around sklearn's IsolationForest to:
      - hide sklearn-specific -1 / +1 labels
      - return 1 for anomaly, 0 for normal
      - return scores where higher means more anomalous
    """

    def __init__(self, contamination=0.05, random_state=None):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit_predict(self, X):
        """
        Fit the model on X and return (preds, scores).

        X : 2D numpy array / sparse matrix.

        preds  : np.ndarray, 1 for anomaly, 0 for normal.
        scores : np.ndarray, anomaly scores (higher = more anomalous).
        """
        self.model.fit(X)

        # decision_function: higher -> more normal
        # we invert sign: higher -> more anomalous
        scores = -self.model.decision_function(X)

        raw_preds = self.model.predict(X)  # -1 for anomaly, +1 for normal
        preds = np.where(raw_preds == -1, 1, 0)

        return preds, scores


class AutoencoderDetector:
    """
    Dense Autoencoder for anomaly detection using reconstruction error.

    Steps:
      1. Train autoencoder to reconstruct feature vectors.
      2. Compute reconstruction MSE for each sample.
      3. Set threshold at (1 - contamination) quantile of errors.
      4. Samples with error > threshold are labeled as anomalies.

    This version is made as deterministic as possible:
      - Seeds numpy + tf with random_state
      - Disables shuffle in model.fit
    """

    def __init__(
        self,
        encoding_dim=32,
        epochs=20,
        batch_size=32,
        contamination=0.05,
        random_state=42,
    ):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    # ------------------------------------------------------------------
    # Utility: force X into dense 2D float32 for Keras
    # ------------------------------------------------------------------
    def _ensure_dense_2d(self, X):
        """
        Convert input X to a dense 2D float32 numpy array that Keras accepts.
        Handles sparse matrices and weird wrappers.
        """
        # If it's a sparse matrix (scipy), convert to dense
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Convert to numpy array
        X = np.asarray(X)

        # If somehow it's 1D, fix to 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Final type for Keras
        return X.astype("float32")

    # ------------------------------------------------------------------
    # Model builder: use Sequential API
    # ------------------------------------------------------------------
    def _build_model(self, input_dim: int):
        """
        Builds a simple fully-connected autoencoder using Sequential API.

        Input -> Dense(encoding_dim, relu) -> Dense(input_dim, linear)
        """
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(self.encoding_dim, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def fit_predict(self, X):
        """
        Train the autoencoder on X and return (preds, scores):

        preds  : 1 for anomaly, 0 for normal
        scores : reconstruction MSE (higher = more anomalous)
        """
        # Make sure X is a clean dense array
        X = self._ensure_dense_2d(X)

        # 🔑 Seed everything for repeatability
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)

        input_dim = X.shape[1]
        self.model = self._build_model(input_dim)

        # Train to reconstruct the input itself
        self.model.fit(
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,  # 🔑 no random shuffling -> more deterministic
            verbose=0,
        )

        # Reconstruction
        recon = self.model.predict(X, verbose=0)
        mse = np.mean((X - recon) ** 2, axis=1)

        # Threshold based on contamination
        threshold = float(np.quantile(mse, 1.0 - self.contamination))

        preds = (mse > threshold).astype(int)  # 1 = anomaly

        return preds, mse
