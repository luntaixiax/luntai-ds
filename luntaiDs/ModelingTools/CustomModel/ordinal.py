import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.preprocessing import label_binarize, LabelEncoder
from scipy.special import softmax

from luntaiDs.ModelingTools.FeatureEngineer.transformers import WeightedAverager
from luntaiDs.ModelingTools.utils.checks import check_params
from luntaiDs.ModelingTools.utils.parallel import parallel_run, delayer


def target_binarize(y, classes) -> pd.DataFrame:
    y_bin = label_binarize(y, classes=classes)
    idmax = y_bin.argmax(axis=1)
    for row_idx, idmx in enumerate(idmax):
        y_bin[row_idx, idmx:] = 1
    return pd.DataFrame(y_bin, columns=classes)


class AlwaysConstEstimator(BaseEstimator):
    def __init__(self, const):
        self.const = const

    def fit(self, X, y, **params):
        return self

    def predict(self, X):
        shape = X.shape
        r = np.zeros(shape[0])
        r[:] = self.const
        return r

    def predict_proba(self, X):
        shape = X.shape
        r = np.ones(shape[0])
        return r


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, ordered_classes: list = None, n_jobs: int = -1):
        """ordinal classification problem
        will decompose the problem into several binary classification problem and calculate the diff probs for each clf
        and use argmax of class prob to predict

        :param estimator:base binary classifier to use
        :param ordered_classes: specify y labels rank from least to most
        :param n_jobs:
        """
        self.estimator = estimator
        self.ordered_classes = np.asarray(ordered_classes)
        self.n_jobs = n_jobs

    @property
    def classes_(self):
        return self.ordered_classes

    @property
    def binary_estimators_(self):
        return self._bin_ests

    def fit(self, X, y, **fit_params):

        y_bins = target_binarize(y, classes=self.ordered_classes)

        self._bin_ests = {}
        # add last classifier (always 1)
        self._bin_ests[self.ordered_classes[-1]] = AlwaysConstEstimator(const=1).fit(X, y)

        # support multi-processor running
        jobs = (self._fit_one(X, y_bins.loc[:, cl], **fit_params) for cl in self.ordered_classes[:-1])
        clfs = parallel_run(jobs, n_jobs=self.n_jobs)
        self._bin_ests.update({cl: clf for cl, clf in zip(self.ordered_classes[:-1], clfs)})

        return self

    @delayer
    def _fit_one(self, X, y_bin, **fit_params):
        clf = clone(self.estimator)
        clf.fit(X, y_bin, **fit_params)
        return clf

    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        return pd.DataFrame(probas, columns=self.classes_).idxmax(axis=1).values

    @delayer
    def _predict_prob_one(self, cl, clf, X):
        prob = clf.predict_proba(X)
        if len(prob.shape) == 2:
            prob = prob[:, 1]  # binary classification, use second column
        return cl, prob  # make sure the class and tree prob is linked correctly

    def predict_proba(self, X) -> np.ndarray:
        diff_probs = self.decision_function(X)
        return softmax(diff_probs, axis=1)

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def decision_function(self, X) -> np.ndarray:
        jobs = (self._predict_prob_one(cl, clf, X) for cl, clf in self.binary_estimators_.items())
        tuples = parallel_run(jobs, n_jobs=self.n_jobs)  # [(cl1, clf1), (cl2, clf2), (cl3, clf3), ...]
        probs = dict(tuples)

        cum_probs = pd.DataFrame(probs)[self.ordered_classes]
        diff_probs = cum_probs - cum_probs.shift(periods=1, axis=1).fillna(0)
        if isinstance(diff_probs, pd.DataFrame):
            return diff_probs.values
        else:
            return diff_probs

    def score(self, X, y, sample_weight=None) -> float:
        # returns kendal tau correlation coefficient
        from scipy import stats
        y_predict = self.predict(X)
        k, p = stats.kendalltau(y, y_predict, nan_policy='omit')
        return k


class OrdinalClassifierPlus(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, ordered_classes: list = None, output_layer: str = 'softmax', n_jobs: int = -1):
        """ordinal classification problem
        will decompose the problem into several binary classification problem and calculate the diff probs for each clf

        :param estimator:base binary classifier to use
        :param ordered_classes: specify y labels rank from least to most
        :param output_layer: {'softmax', 'lda', 'qda', 'wa+lda', 'wa+qda'}, will be applied after each clf output a prob to get the final decision (stacking)
                -- softmax: will choose the argmax prob of each class
                -- lda/qda: use lda or qda to make final decision
                -- wa+lda/wa+qda: add weighted average before feeding into lda/qda, this way will be uni-variate X and y classification
        :param n_jobs:
        """
        self.estimator = estimator
        self.ordered_classes = np.asarray(ordered_classes)
        self.output_layer = check_params(output_layer, allowed_values=['softmax', 'lda', 'qda', 'wa+lda', 'wa+qda'])
        self.n_jobs = n_jobs

        # stacking will automatically labelize y, which will conflict our base estimator, so need to manually map y to ints and map back when predict
        self._label_enc = LabelEncoder().fit(ordered_classes)
        if output_layer == 'softmax':
            final_est = LogisticRegression()
        elif output_layer == 'lda':
            final_est = LinearDiscriminantAnalysis(store_covariance=True)
        elif output_layer == 'qda':
            final_est = QuadraticDiscriminantAnalysis(store_covariance=True)
        elif output_layer == 'wa+lda':
            k = len(ordered_classes)
            final_est = Pipeline([
                ('wa', WeightedAverager(weights=range(1, k + 1), intercept=0.0, n_jobs=n_jobs)),
                ('lda', LinearDiscriminantAnalysis(store_covariance=True))]
            )
        elif output_layer == 'wa+qda':
            k = len(ordered_classes)
            final_est = Pipeline([
                ('wa', WeightedAverager(weights=range(1, k + 1), intercept=0.0, n_jobs=n_jobs)),
                ('qda', QuadraticDiscriminantAnalysis(store_covariance=True))]
            )

        self.stack_ = StackingClassifier(
            estimators=[('ordinal',
                         OrdinalClassifier(estimator, ordered_classes=self._label_enc.transform(ordered_classes),
                                           n_jobs=n_jobs))],
            final_estimator=final_est,
            stack_method='decision_function',  # use decision function of the base estimators
            n_jobs=n_jobs,
        )

    @property
    def base_ordinal_estimator_(self):
        return self.stack_.named_estimators_['ordinal']

    @property
    def final_estimator_(self):
        return self.stack_.final_estimator_

    @property
    def feature_names_in_(self):
        return self.stack_.feature_names_in_

    @property
    def classes_(self):
        return self.ordered_classes

    @property
    def binary_estimators_(self):
        return self.base_ordinal_estimator_._bin_ests

    def fit(self, X, y, **fit_params):
        y_labled = self._label_enc.transform(y)
        self.stack_.fit(X, y_labled, **fit_params)
        return self

    def predict(self, X):
        y = self.stack_.predict(X)
        # map back encoded y to original y space
        return self._label_enc.inverse_transform(y)

    def predict_proba(self, X):
        # need to follow original sequence of classes_
        p = self.stack_.predict_proba(X)
        classes_ = self._label_enc.inverse_transform(self.stack_.classes_)
        p = pd.DataFrame(p, columns=classes_)[self.classes_]
        return p.values

    def decision_function(self, X):
        # need to follow original sequence of classes_
        p = self.stack_.decision_function(X)
        classes_ = self._label_enc.inverse_transform(self.stack_.classes_)
        p = pd.DataFrame(p, columns=classes_)[self.classes_]
        return p.values

    def score(self, X, y, sample_weight=None):
        # returns kendal tau correlation coefficient
        from scipy import stats
        y_predict = self.predict(X)
        k, p = stats.kendalltau(y, y_predict, nan_policy='omit')
        return k
