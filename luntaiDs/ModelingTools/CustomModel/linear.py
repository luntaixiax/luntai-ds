import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.metrics import get_scorer
from sklearn.preprocessing import LabelEncoder


# https://www.statsmodels.org/stable/regression.html#
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
# https://stats.stackexchange.com/questions/16493/difference-between-confidence-intervals-and-prediction-intervals
# https://stats.stackexchange.com/questions/136157/general-mathematics-for-confidence-interval-in-multiple-linear-regression

class _linearStatsModelWrapper(BaseEstimator):
    def __init__(self, model_family: str = 'glm', fit_intercept: bool = False, model_params: dict = None,
                 fit_params: dict = None, scoring: str = None):
        self.model_family = model_family
        self.fit_intercept = fit_intercept
        self.model_params = {} if model_params is None else model_params
        self.fit_params = {} if fit_params is None else fit_params
        self.scoring = scoring
        self.le = None

    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns
        model_cls = {
            # continious regression model
            'ols': sm.OLS,
            'gls': sm.GLS,
            'wls': sm.WLS,
            'rlm': sm.RLM,
            # discrete classification model
            'glm': sm.GLM,
            'logit': sm.Logit,
            'probit': sm.Probit,
            'poisson': sm.Poisson,
        }.get(self.model_family, sm.GLM)

        if self.fit_intercept:
            X = sm.add_constant(X)
        if self.le is not None:
            self.le = self.le.fit(y)
            y = self.le.transform(y)

        model = model_cls(y, X, **self.model_params)
        self.model_result = model.fit(freq_weight=sample_weight, **self.fit_params)

        return self

    def score(self, X, y, sample_weight=None):
        if callable(self.scoring):
            s = self.scoring(self, X, y)
            return s

        try:
            s = getattr(self.model_result_, self.scoring)
        except AttributeError:
            if isinstance(self.scoring, str):
                scorer = get_scorer(self.scoring)
                s = scorer(self, X, y)
            else:
                raise TypeError("Scoring must be str or callable")
        return s

    @property
    def model_result_(self):
        return self.model_result


class LinearClfStatsModelWrapper(_linearStatsModelWrapper, ClassifierMixin):
    def __init__(self, model_family: str = 'glm', fit_intercept: bool = False, model_params: dict = None,
                 fit_params: dict = None, scoring: str = 'roc_auc'):
        if model_family in ('ols', 'gls', 'wls', 'rlm'):
            raise ValueError(
                "the model_family falls in the regresion type model, please use LinearRegStatsModelWrapper class instead")
        super().__init__(model_family=model_family, fit_intercept=fit_intercept, model_params=model_params,
                         fit_params=fit_params, scoring=scoring)
        self.le = LabelEncoder()

    def decision_function(self, X) -> np.ndarray:
        if len(self.classes_) > 2:
            return self.predict_proba(X)
        else:
            return self.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        if self.fit_intercept:
            X = sm.add_constant(X)
        probs = self.model_result_.predict(X).values
        return np.c_[1 - probs, probs]

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.5, 1, 0)

    @property
    def classes_(self):
        return self.le.classes_

    @property
    def coef_(self):
        coefs = self.model_result_.params
        if self.fit_intercept:
            return coefs.values[1:]  # 1st is intercept
        else:
            return coefs.values

    @property
    def intercept_(self):
        coefs = self.model_result_.params
        if self.fit_intercept:
            return np.array([coefs['const']])
        else:
            return np.array([0.0])


class LinearRegStatsModelWrapper(_linearStatsModelWrapper, RegressorMixin):
    def __init__(self, model_family: str = 'ols', fit_intercept: bool = False, model_params: dict = None,
                 fit_params: dict = None, scoring: str = 'r2'):
        if model_family not in ('ols', 'gls', 'wls', 'rlm'):
            raise ValueError(
                "the model_family falls in the classification type model, please use LinearClfStatsModelWrapper class instead")
        super().__init__(model_family=model_family, fit_intercept=fit_intercept, model_params=model_params,
                         fit_params=fit_params, scoring=scoring)

    def predict(self, X) -> np.ndarray:
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.model_result_.predict(X).values