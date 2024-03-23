from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import is_classifier
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.model_selection import check_cv
from skopt import BayesSearchCV
from optuna import logging
from optuna import samplers
from optuna import study as study_module
from optuna.integration.sklearn import OptunaSearchCV, _logger, _num_samples, _safe_indexing, _Objective, _check_fit_params, _convert_old_distribution_to_new_distribution
from optuna.trial import Trial

from luntaiDs.ModelingTools.utils.support import make_present_col_selector


def convert_subset_clf(base_estimator: BaseEstimator, features_subset: list) -> Pipeline:
    c = ColumnTransformer(
        transformers = [('subset', 'passthrough', make_present_col_selector(features_subset))], # pass all features show up in features_subset, no transformation is made, so 'passthrough'
        remainder = 'drop' # drop all remaining columns not presented in features_subset
    )
    pipe = Pipeline([
        ('transform', c),
        ('model', base_estimator)
    ])
    return pipe

class GroupStratifiedBayesSearchCV(BayesSearchCV):
    def __init__(self, estimator, group_col: str, search_spaces, n_cv: int = 3, optimizer_kwargs=None, n_iter=50,
                 scoring=None,
                 fit_params=None, n_jobs=1, n_points=1, refit=True, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None, error_score='raise', return_train_score=False):
        self.group_col = group_col
        self.n_cv = n_cv
        super().__init__(estimator=estimator, search_spaces=search_spaces, optimizer_kwargs=optimizer_kwargs,
                         n_iter=n_iter, scoring=scoring, fit_params=fit_params, n_jobs=n_jobs, n_points=n_points,
                         refit=refit, cv=StratifiedGroupKFold(n_splits=n_cv, random_state=random_state),
                         verbose=verbose, pre_dispatch=pre_dispatch,
                         random_state=random_state, error_score=error_score, return_train_score=return_train_score)

    def fit(self, X, y=None, *, groups=None, callback=None, **fit_params):
        if self.group_col not in X.columns:
            raise KeyError(f"{self.group_col} should be in training X")

        X_ = X.drop(columns=self.group_col)
        kfold_groups = X[self.group_col]
        return super().fit(X_, y, groups=kfold_groups, callback=callback, **fit_params)

    def score(self, X, y=None):
        return super().score(X.drop(columns=self.group_col), y)

    def score_samples(self, X):
        return super().score_samples(X.drop(columns=self.group_col))

    def predict(self, X):
        return super().predict(X.drop(columns=self.group_col))

    def predict_proba(self, X):
        return super().predict_proba(X.drop(columns=self.group_col))

    def predict_log_proba(self, X):
        return super().predict_log_proba(X.drop(columns=self.group_col))

    def decision_function(self, X):
        return super().decision_function(X.drop(columns=self.group_col))

    def transform(self, X):
        return super().transform(X.drop(columns=self.group_col))

class GroupStratifiedOptunaSearchCV(OptunaSearchCV):
    def __init__(self, estimator, group_col: str, param_distributions, n_cv: int = 3, enable_pruning=False, error_score=np.nan,
                 max_iter=1000, n_jobs=1, n_trials=10, random_state=None, refit=True, return_train_score=False,
                 scoring=None, study=None, subsample=1.0, timeout=None, verbose=0, callbacks=None):
        if not isinstance(param_distributions, dict):
            raise TypeError("param_distributions must be a dictionary.")
        self.group_col = group_col
        self.n_cv = n_cv

        self.cv = StratifiedGroupKFold(n_splits=n_cv, random_state=random_state)
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.estimator = estimator
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.study = study
        self.subsample = subsample
        self.timeout = timeout
        self.verbose = verbose
        self.callbacks = callbacks


    def fit(self, X, y=None, groups=None, **fit_params):
        if self.group_col not in X.columns:
            raise KeyError(f"{self.group_col} should be in training X")
        X_ = X.drop(columns=self.group_col)
        kfold_groups = X[self.group_col]

        # TODO: moved from __init__ to here, check if there are negative implications
        self.param_distributions = {
            key: _convert_old_distribution_to_new_distribution(dist)
            for key, dist in self.param_distributions.items()
        }
        return super().fit(X_, y, groups=kfold_groups, **fit_params)

    def score(self, X, y=None):
        return super().score(X.drop(columns=self.group_col), y)

    def predict(self, X):
        return super().predict(X.drop(columns=self.group_col))

    def predict_proba(self, X):
        return super().predict_proba(X.drop(columns=self.group_col))

    def predict_log_proba(self, X):
        return super().predict_log_proba(X.drop(columns=self.group_col))

    def decision_function(self, X):
        return super().decision_function(X.drop(columns=self.group_col))

    def transform(self, X):
        return super().transform(X.drop(columns=self.group_col))