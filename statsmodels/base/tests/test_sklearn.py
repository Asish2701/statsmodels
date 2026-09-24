import numpy as np
import pytest

sm = pytest.importorskip("statsmodels.api")
pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.base.sklearn import SMClassifier, SMRegressor


def test_regressor_sklearn_interface():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(40, 3))
    y = 1 + 2 * X[:, 0] - X[:, 1] + rng.normal(size=40, scale=0.1)

    estimator = SMRegressor(sm.OLS)
    assert estimator.get_params() == {
        "estimator": sm.OLS,
        "fit_intercept": True,
        "model_kwargs": None,
        "fit_kwargs": None,
    }
    assert clone(estimator).get_params() == estimator.get_params()

    estimator.fit(X, y)
    np.testing.assert_allclose(estimator.predict(X), estimator.results_.fittedvalues)
    assert estimator.n_features_in_ == 3
    assert estimator.summary() is not None


def test_regressor_pipeline_and_cross_validation():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(60, 2))
    y = 2 + X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=60, scale=0.2)

    estimator = make_pipeline(StandardScaler(), SMRegressor(sm.OLS))
    scores = cross_val_score(estimator, X, y, cv=3)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_regressor_model_and_fit_kwargs():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + rng.normal(size=30)

    estimator = SMRegressor(
        sm.OLS,
        model_kwargs={"missing": "raise"},
        fit_kwargs={"method": "pinv"},
    )
    estimator.fit(X, y)

    assert estimator.results_.params.shape == (3,)


def test_classifier_labels_and_probabilities():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(80, 2))
    y = np.where(X[:, 0] - X[:, 1] > 0, "yes", "no")

    estimator = SMClassifier(sm.Logit, fit_kwargs={"disp": False})
    estimator.fit(X, y)

    assert np.array_equal(estimator.classes_, np.array(["no", "yes"]))
    probabilities = estimator.predict_proba(X)
    assert probabilities.shape == (80, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(estimator.predict(X)) <= {"no", "yes"}


def test_classifier_cross_validation():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    estimator = SMClassifier(sm.Logit, fit_kwargs={"disp": False})
    scores = cross_val_score(estimator, X, y, cv=3)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_classifier_requires_two_classes():
    X = np.arange(10).reshape(5, 2)
    y = np.ones(5)

    with pytest.raises(ValueError, match="binary classification"):
        SMClassifier(sm.Logit, fit_kwargs={"disp": False}).fit(X, y)


def test_classifier_with_pipeline():
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    estimator = make_pipeline(
        StandardScaler(), SMClassifier(sm.Logit, fit_kwargs={"disp": False})
    )
    scores = cross_val_score(estimator, X, y, cv=3)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
