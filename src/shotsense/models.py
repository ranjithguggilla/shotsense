"""Candidate xG models. Every candidate is a scikit-learn style estimator whose
preprocessing lives inside the pipeline so cross-validation cannot leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pygam import LogisticGAM, s
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import BINARY, CATEGORICAL, FREEZE, NUMERIC_BASE, SETS


class GamClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper over a pyGAM logistic additive model on two smooth terms."""

    def __init__(self, n_splines: int = 12, lam: float = 1.0):
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for j in range(1, X.shape[1]):
            terms = terms + s(j, n_splines=self.n_splines, lam=self.lam)
        self.gam_ = LogisticGAM(terms).fit(X, np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.gam_.predict_mu(np.asarray(X, dtype=float))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _columns(features: list[str]):
    num = [c for c in features if c in NUMERIC_BASE or c in FREEZE]
    binary = [c for c in features if c in BINARY]
    cat = [c for c in features if c in CATEGORICAL]
    return num, binary, cat


def _preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    num, binary, cat = _columns(features)
    parts = []
    if num:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=False))]
        if scale:
            steps.append(("scale", StandardScaler()))
        parts.append(("num", Pipeline(steps), num))
    if binary:
        parts.append(("bin", "passthrough", binary))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                           sparse_output=False), cat))
    return ColumnTransformer(parts)


def build_model(name: str, seed: int = 7) -> Pipeline:
    if name == "logit_distance":
        feats = SETS["distance"]
        return Pipeline([("prep", _preprocessor(feats, True)),
                         ("clf", LogisticRegression(C=1e4, max_iter=1000))])
    if name == "logit_geometry":
        feats = ["distance", "angle", "log_distance"]
        return Pipeline([("prep", _preprocessor(feats, True)),
                         ("clf", LogisticRegression(C=1e4, max_iter=1000))])
    if name == "gam_geometry":
        feats = ["distance", "angle"]
        return Pipeline([("prep", _preprocessor(feats, True)), ("clf", GamClassifier())])
    if name == "logit_context":
        feats = SETS["context"]
        return Pipeline([("prep", _preprocessor(feats, True)),
                         ("clf", LogisticRegression(C=1.0, max_iter=3000))])
    if name == "boosting_context":
        feats = SETS["context"]
        return Pipeline([("prep", _preprocessor(feats, False)),
                         ("clf", HistGradientBoostingClassifier(
                             learning_rate=0.05, max_iter=300, max_leaf_nodes=12,
                             min_samples_leaf=60, l2_regularization=1.0,
                             early_stopping=True, random_state=seed))])
    if name == "boosting_defenders":
        feats = SETS["context_defenders"]
        return Pipeline([("prep", _preprocessor(feats, False)),
                         ("clf", HistGradientBoostingClassifier(
                             learning_rate=0.05, max_iter=300, max_leaf_nodes=12,
                             min_samples_leaf=60, l2_regularization=1.0,
                             early_stopping=True, random_state=seed))])
    raise KeyError(name)


FEATURES_FOR = {
    "logit_distance": SETS["distance"],
    "logit_geometry": ["distance", "angle", "log_distance"],
    "gam_geometry": ["distance", "angle"],
    "logit_context": SETS["context"],
    "boosting_context": SETS["context"],
    "boosting_defenders": SETS["context_defenders"],
}
CANDIDATES = list(FEATURES_FOR)


def predict(model: Pipeline, name: str, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES_FOR[name]])[:, 1]
