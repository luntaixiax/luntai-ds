import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import LabelEncoder


class IsotonicRegWithInterp(BaseEstimator, ClassifierMixin):
    def __init__(self, y_min: float, y_max: float, increasing: bool = True, interp_method: str = 'pchi', cv: int = 3):
        """Ensemble isotonic calibration while fitting an interpolate function to avoid flat mapping

        :param y_min: min y for Isotonic Regression
        :param y_max: max y for Isotonic Regression
        :param increasing: whether the trend is increasing or decreasing
        :param interp_method: one of {monotonic, cubic, akima}
        :param cv: number of bagging Isotonic regressors to alleviate overfitting on specific dataset
        """
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.interp_method = interp_method
        self.cv = cv

    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        if len(self.le_.classes_) > 2:
            raise ValueError("Only support binary classification problem!")

        self.iso = BaggingRegressor(
            IsotonicRegression(y_min=self.y_min, y_max=self.y_max, increasing=self.increasing, out_of_bounds='clip'),
            n_estimators=self.cv
        )
        self.iso.fit(X, y)
        self.X_thresholds = np.sort(np.unique(np.concatenate([t.X_thresholds_ for t in self.iso.estimators_])))
        self.y_thresholds = self.iso.predict(self.X_thresholds_[:, np.newaxis])

        x = np.concatenate(([0], (self.X_thresholds[1::2] + self.X_thresholds[:-1:2]) / 2, [1]))
        z = np.concatenate(([0], (self.y_thresholds[1::2] + self.y_thresholds[:-1:2]) / 2, [1]))
        if self.interp_method == 'monotonic':
            self.interp = PchipInterpolator(x, z, extrapolate=True)
        elif self.interp_method == 'cubic':
            self.interp = CubicSpline(x, z, extrapolate=True)
        elif self.interp_method == 'akima':
            self.interp = Akima1DInterpolator(x, z)

    @property
    def classes_(self):
        return self.le_.classes_

    @property
    def isobagger_(self):
        return self.iso

    @property
    def interp_(self):
        return self.interp

    @property
    def X_thresholds_(self):
        return self.X_thresholds

    @property
    def y_thresholds_(self):
        return self.y_thresholds

    def predict(self, X) -> np.ndarray:
        threshold = 0.5
        return np.where(self.predict_proba(X)[:, 1] >= threshold, 1, 0)

    def predict_proba(self, X) -> np.ndarray:
        return np.c_[1 - self.interp(X), self.interp(X)]

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def decision_function(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X)[:, 1])