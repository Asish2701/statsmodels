Scikit-Learn Compatibility
===========================

statsmodels provides optional wrappers for using selected statsmodels models
with the scikit-learn estimator interface.

The wrappers are available in :mod:`statsmodels.base.sklearn` after installing
the optional scikit-learn dependency. For example:

.. code-block:: bash

   pip install statsmodels[sklearn]

A regression model can be used with scikit-learn model selection and pipeline
utilities while retaining the fitted statsmodels results object:

.. code-block:: python

   import statsmodels.api as sm
   from statsmodels.base.sklearn import SMRegressor

   model = SMRegressor(sm.OLS)
   model.fit(X, y)

   predictions = model.predict(X_test)
   summary = model.summary()

The fitted statsmodels results are available through ``results_``. Model
constructor options can be supplied through ``model_kwargs`` and options for
the statsmodels ``fit`` method through ``fit_kwargs``.

``SMClassifier`` provides the same interface for binary classification models
such as ``statsmodels.api.Logit``. It exposes ``classes_``, ``predict`` and
``predict_proba`` using the original target labels.

.. note::

   The wrappers currently support binary classification. The statsmodels
   estimator passed to ``SMClassifier`` must return a one-dimensional vector
   of probabilities from ``predict``.
