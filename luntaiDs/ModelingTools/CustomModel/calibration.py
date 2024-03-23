import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator
from scipy.special import logit, expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import LabelEncoder


class IsotonicRegWithInterp(BaseEstimator, ClassifierMixin):
    def __init__(self, y_min: float, y_max: float, increasing: bool = True, interp_method: str = 'monotonic', cv: int = 3):
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
    

def calc_realized_event_rt(y_pred: np.ndarray, y_true: np.ndarray, cutoffs: np.ndarray, labels: list = None) -> pd.Series:
    """realized event rate within each cutoff bucket

    :param np.ndarray y_pred: 1d original scores by the model (continuous)
    :param np.ndarray y_true: 1d binary event variable (0,1)
    :param np.ndarray cutoffs: given cutoff points, at scores (y_pred) scale
    :param list labels: if given, use to give output
    :return _type_: _description_
    """
    if labels is None:
        labels = [f"LB{i:03d}" for i in range(len(cutoffs) - 1)]
    df = pd.DataFrame({
        'buckets' : pd.cut(
            y_pred, 
            bins = cutoffs, 
            right = True, 
            labels = labels
        ).astype('object'),
        'event' : y_true
    })
    return df.groupby('buckets')['event'].mean()


class MappingCutoffCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, eval_prob_cutoff: np.ndarray, proposed_score_cutoff: np.ndarray = None, 
                interp_method: str = 'monotonic', prob_logit:bool = True) -> None:
        """binary calibrator that aims to let realized event rate fall within given eval_cutoff buckets

        your input:
            1. feature X and binary target y, 
            2. a trained binay classifier clf, which will output score  s = clf(X)
            3. n ranking (bucket) system with probability (or transformed one) cutoff points [p0, p1, p2, ..., pn]
                bucket    lower_bound    upper_bound
                1         p0             p1
                2         p1             p2
                3         p2             p3
                ...
                n         pn-1           pn

        your goal:
            find the mapping function f that map the score s into probability p: p = f(s)
            the mapping function is parameterized by a series of cutoff points [s0, s1, s2, ..., sn], 
                and is constructed by interplotation of two set of cutoff points s and p:
                p = f(s) = inter(s, [s0, s1, s2, ..., sn], [p0, p1, p2, ..., pn])

            the best mapping function is to find the combination of [s0, s1, s2, ..., sn] that makes realized event rate
            within each s bucket fall into p boundaries:
                bucket    lower_bound    upper_bound   realized_event_rt
                1         s0             s1            q1
                2         s1             s2            q2
                3         s2             s3            q3
                ...
                n         sn-1           sn            qn
            so that pi-1 <= qi <= pi
        
        distortion function:
            for better fitting, you may need to convert both score and probability cutoff into linear space
            s' = score_transform_func(s), usually logit (logodd) transformation, can add in pipeline in previous step
            p' = logit(p), p = expit(p')
            and [p0', p1', p2', ..., pn'] should also be transformed as well

            so that during transformation step.  s' --f-> p' -> p   ~  y

        :param np.ndarray eval_prob_cutoff: evaluation probability cutoff points, [p0, p1, p2, ..., pn] on prob scale
        :param np.ndarray proposed_score_cutoff: if given, will use this as [s0', s1', s2', ..., sn'], i.e., manual mode
                if not, it will be the target of the fit function
        :param str interp_method: interpolate method for mapping function f, one of {monotonic, cubic, akima}
        :param bool prob_logit: whether to do logit transform on p: p -> p' so that it will be on linear space
        """
        self.eval_prob_cutoff = eval_prob_cutoff
        self.proposed_score_cutoff = proposed_score_cutoff
        self.interp_method = interp_method
        self.prob_logit = prob_logit

    def fit(self, X, y, **fit_params):
        # X only have one column - score to be calibrated
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        if len(self.le_.classes_) > 2:
            raise ValueError("Only support binary classification problem!")

        # convert X to pandas
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] > 1:
            raise ValueError("Only support one single feature - score")

        
        if self.proposed_score_cutoff is not None:
            self.score_cutoff = np.array(self.proposed_score_cutoff)
        else:
            # try combinations of cutoffs and find the best one
            pass

        # fit interpolator:
        sc = self.score_cutoff_.clip(min = -1e9, max = 1e9)
        if self.prob_logit:
            ec = logit(np.clip(self.eval_prob_cutoff, 1e-9, 1 - 1e-9))
        else:
            ec = self.eval_prob_cutoff
        if self.interp_method == 'monotonic':
            self.interp = PchipInterpolator(sc, ec, extrapolate=True)
        elif self.interp_method == 'cubic':
            self.interp = CubicSpline(sc, ec, extrapolate=True)
        elif self.interp_method == 'akima':
            self.interp = Akima1DInterpolator(sc, ec)

        self.pipe = Pipeline([
            ('interp', FunctionTransformer(func = self.interp)),
            ('prob_transform', FunctionTransformer(func = expit if self.prob_logit else None))
        ]).fit(X)

    def predict_proba(self, X) -> np.ndarray:
        # X only have one column - score to be calibrated
        z = self.pipe.transform(X)[:, 0]
        return np.c_[1 - z, z]

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def decision_function(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X)[:, 1])

    def realized_by_bucket(self, X, y, cutoffs = None) -> pd.DataFrame:
        if cutoffs is None:
            cutoffs = self.score_cutoff_
        # convert X to pandas
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] > 1:
            raise ValueError("Only support one single feature - score")
    

        labels = [f"LB{i:03d}" for i in range(len(cutoffs) - 1)]
        rb = pd.merge(
            calc_realized_event_rt(
                y_pred = X.values[:,0],
                y_true = self.le_.transform(y),
                cutoffs = cutoffs,
                labels = labels
            ),
            pd.merge(
                pd.Series(
                    self.eval_prob_cutoff[:-1], 
                    index = labels, 
                    name = 'lower_bound'
                ),
                pd.Series(
                    self.eval_prob_cutoff[1:], 
                    index = labels, 
                    name = 'upper_bound'
                ),
                how = 'outer',
                left_index = True,
                right_index = True
            ),
            how = 'outer',
            left_index = True,
            right_index = True
        )
        return rb


    @property
    def classes_(self):
        return self.le_.classes_

    @property
    def score_cutoff_(self) -> np.ndarray:
        return self.score_cutoff