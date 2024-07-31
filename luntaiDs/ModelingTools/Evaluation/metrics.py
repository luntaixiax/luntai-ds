from enum import Enum
from typing import Dict, Literal
import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, \
        average_precision_score, f1_score, log_loss, precision_score, recall_score, \
        roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics._ranking import _binary_clf_curve

def ks_stat(y, yhat):
    return ks_2samp(yhat[y==1], yhat[y!=1]).statistic

def binary_clf_metric_at_thresholds(y_true: np.ndarray, pred_prob: np.ndarray) -> pd.DataFrame:
    """Binary classification metrics at each threshold

    :param y_true: ground truth target array (binary), 1-d array
    :param pred_prob: predicted probabilities array (float between 0-1), 1-d array
    :return: DataFrame of columns [threshold, tp, fp, tn, fn, tpr, fpr, precision, recall]
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, pred_prob)
    num_p = y_true.sum()
    num_n = (1 - y_true).sum()

    t = pd.DataFrame({
        'threshold': thresholds,
        'tp': tps,
        'fp': fps,
        'tn': num_n - fps,
        'fn': num_p - tps
    })
    t['tpr'] = t['tp'] / (t['tp'] + t['fn'])
    t['fpr'] = t['fp'] / (t['fp'] + t['tn'])
    t['precision'] = t['tp'] / (t['tp'] + t['fp'])
    t['recall'] = t['tpr']
    return t



class _SKMetricsMultiCls:
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        """a standardized sklearn scorer, support for multi-class

        :param np.ndarray y_true: ground truth, 1-d array, size (n,)
        :param np.ndarray y_pred_probs: 2-d array, size (n, n_class), the predicted probabilities
            note that the order of the columns should be exactly same as labels in ground truth array
        :return float: the metric result
        """
        raise NotImplementedError("")
    
    def _convert_when_binary(self, y_pred_probs: np.ndarray) -> np.ndarray:
        """sklearn metrics are not behaving same for binary vs. multi-class

        :param np.ndarray y_pred_probs: predicted probabilities, size (n, n_class)
        :return np.ndarray: if binary case, extracted probs for class 1, size (n, )
                            if multi-class case, do nothing, size (n, n_class)
        """
        if y_pred_probs.shape[1] == 2:
            if isinstance(y_pred_probs, pd.DataFrame):
                return y_pred_probs.iloc[:, 1]
            return y_pred_probs[:, 1]
        return y_pred_probs
        

class Accurary(_SKMetricsMultiCls):
    def __init__(self, balanced: bool = False, top_k: int = 1):
        if balanced and top_k != 1:
            logging.warning('if set to balanced, will not consider top_k param')
        self._balanced = balanced
        self._topk = top_k
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        if self._balanced:
            y_pred_labels = np.argmax(y_pred_probs, axis = 1)
            return balanced_accuracy_score(y_true, y_pred_labels)
        if self._topk == 1:
            y_pred_labels = np.argmax(y_pred_probs, axis = 1)
            return accuracy_score(y_true, y_pred_labels)
        if self._topk != 1:
            y_pred_probs = self._convert_when_binary(y_pred_probs)
            return top_k_accuracy_score(y_true, y_pred_probs, k = self._topk)
    
class AvgPrecision(_SKMetricsMultiCls):
    def __init__(self, average: Literal['micro', 'macro', 'weighted']):
        self._average = average
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_true_binary = label_binarize(y_true, classes = list(range(y_pred_probs.shape[1])))
        y_pred_probs = self._convert_when_binary(y_pred_probs)
        return average_precision_score(y_true_binary, y_pred_probs, average = self._average)
    
class F1Score(_SKMetricsMultiCls):
    def __init__(self, average: Literal['micro', 'macro', 'weighted']):
        self._average = average
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_pred_labels = np.argmax(y_pred_probs, axis = 1)
        return f1_score(y_true, y_pred_labels, average = self._average)
    
class Precision(_SKMetricsMultiCls):
    def __init__(self, average: Literal['micro', 'macro', 'weighted']):
        self._average = average
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_pred_labels = np.argmax(y_pred_probs, axis = 1)
        return precision_score(y_true, y_pred_labels, average = self._average)
    
class Recall(_SKMetricsMultiCls):
    def __init__(self, average: Literal['micro', 'macro', 'weighted']):
        self._average = average
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_pred_labels = np.argmax(y_pred_probs, axis = 1)
        return recall_score(y_true, y_pred_labels, average = self._average)
    
class RocAuc(_SKMetricsMultiCls):
    def __init__(self, average: Literal['micro', 'macro', 'weighted']):
        self._average = average
        
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_true_binary = label_binarize(y_true, classes = list(range(y_pred_probs.shape[1])))
        y_pred_probs = self._convert_when_binary(y_pred_probs)
        return roc_auc_score(y_true_binary, y_pred_probs, 
                            average = self._average)
    
class EntropyLoss(_SKMetricsMultiCls):
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        y_pred_probs = self._convert_when_binary(y_pred_probs)
        return log_loss(y_true, y_pred_probs)


class SKClfMetricsEnum(Enum):
    accuracy = Accurary(balanced=False)
    balanced_accuracy = Accurary(balanced=True)
    top_2_accuracy = Accurary(top_k=2)
    top_5_accuracy = Accurary(top_k=5)
    average_precision_micro = AvgPrecision(average='micro')
    average_precision_macro = AvgPrecision(average='macro')
    average_precision_weighted = AvgPrecision(average='weighted')
    f1_micro = F1Score(average='micro')
    f1_macro = F1Score(average='macro')
    f1_weighted = F1Score(average='weighted')
    precision_micro = Precision(average='micro')
    precision_macro = Precision(average='macro')
    precision_weighted = Precision(average='weighted')
    recall_micro = Recall(average='weighted')
    recall_macro = Recall(average='macro')
    recall_weighted = Recall(average='weighted')
    roc_auc_micro = RocAuc(average='micro')
    roc_auc_macro = RocAuc(average='macro')
    roc_auc_weighted = RocAuc(average='weighted')
    neg_log_loss = EntropyLoss()
    
    
    def __call__(self, y_true: np.ndarray, y_pred_probs: np.ndarray):
        return self.value(y_true, y_pred_probs)
    
    
class SKMultiClfMetricsCalculator:
    def __init__(self, y_true: np.ndarray, y_pred_probs: np.ndarray):
        """_summary_

        :param np.ndarray y_true: ground truth, 1-d array, size (n,)
        :param np.ndarray y_pred_probs: 2-d array, size (n, n_class), the predicted probabilities
            note that the order of the columns should be exactly same as labels in ground truth array
        """
        self.y_true = y_true
        if isinstance(y_pred_probs, pd.DataFrame):
            self.y_pred_probs = y_pred_probs.values
        else:
            self.y_pred_probs = y_pred_probs
        self.num_cls = y_pred_probs.shape[1]
        
    def score(self, metric: SKClfMetricsEnum) -> float:
        """get classification score metric

        :param SKClfMetricsEnum metric: metric defined by SKClfMetricsEnum
        :return float: metric calculated on the given dataset
        """
        return metric(self.y_true, self.y_pred_probs)
    
    def binary_metrics_by_threshold(self, cls_idx: int) -> pd.DataFrame:
        """get binary classification metrics by threshold (1 vs rest approach)

        :param int cls_idx: the class index
        :return pd.DataFrame: DataFrame of columns 
                threshold, tp, fp, tn, fn, tpr, fpr, precision, recall]
        """
        if 0 <= cls_idx <= self.num_cls - 1:
            y_true_binarized = label_binarize(
                self.y_true, 
                classes = list(range(self.y_pred_probs.shape[1]))
            )
            y_true = y_true_binarized[:, cls_idx]
            pred_prob = self.y_pred_probs[:, cls_idx]
            return binary_clf_metric_at_thresholds(y_true, pred_prob)
        else:
            raise ValueError(f"Total {self.num_cls} classes")
        
    def roc_auc_curves(self) -> Dict[int, pd.DataFrame]:
        """roc auc curves

        :return Dict[int, pd.DataFrame]: {
                cls_idx: DataFrame[threshold, tpr, fpr]
            }
        """
        curves = dict()
        for cls_idx in range(self.num_cls):
            metrics = self.binary_metrics_by_threshold(cls_idx)
            curves[cls_idx] = metrics[['threshold', 'tpr', 'fpr']]
        return curves
    
    def roc_auc_curve_agg(self, agg: Literal['micro', 'macro']) -> pd.DataFrame:
        """get averaged roc auc (tpr-fpr)
        
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#one-vs-rest-multiclass-roc
        
        In a multi-class classification setup with highly imbalanced classes, 
            micro-averaging is preferable over macro-averaging

        :param Literal[&#39;micro&#39;, &#39;macro&#39;] agg: aggregation method
        :return pd.DataFrame: containing [threshold, tpr, fpr]
        """
        y_true_binarized = label_binarize(
            self.y_true, 
            classes = list(range(self.y_pred_probs.shape[1]))
        )
        if agg == 'micro':
            fpr, tpr, threshold = roc_curve(
                y_true_binarized.ravel(), 
                self.y_pred_probs.ravel(),
                drop_intermediate=False
            )

        elif agg == 'macro':
            fprs, tprs, thresholds = dict(), dict(), dict()
            for i in range(self.num_cls):
                fprs[i], tprs[i], thresholds[i] = roc_curve(
                    y_true_binarized[:, i], 
                    self.y_pred_probs[:, i],
                    drop_intermediate=False
                )
    
            threshold = np.linspace(0.0, 1.0, 2500)
            tpr = np.zeros_like(threshold)
            fpr = np.zeros_like(threshold)
            for i in range(self.num_cls):
                # linear interpolation
                # need to be reversed order because second param must be increasing
                tpr += np.interp(threshold, thresholds[i][::-1], tprs[i][::-1]) 
                fpr += np.interp(threshold, thresholds[i][::-1], fprs[i][::-1])
                
            # Average it and compute AUC
            tpr /= self.num_cls
            fpr /= self.num_cls
            
        else:
            raise ValueError("only support micro or macro aggregation")
        
        return pd.DataFrame({
                'threshold': threshold,
                'fpr': fpr,
                'tpr': tpr
            })
        
    def precision_recall_curves(self) -> Dict[int, pd.DataFrame]:
        """precision recall curves

        :return Dict[int, pd.DataFrame]: {
                cls_idx: DataFrame[threshold, precision, recall]
            }
        """
        curves = dict()
        for cls_idx in range(self.num_cls):
            metrics = self.binary_metrics_by_threshold(cls_idx)
            curves[cls_idx] = metrics[['threshold', 'precision', 'recall']]
        return curves
        
    def precision_recall_curve_agg(self, agg: Literal['micro', 'macro']) -> pd.DataFrame:
        """get averaged precision-recall curve
        
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#one-vs-rest-multiclass-roc
        
        In a multi-class classification setup with highly imbalanced classes, 
            micro-averaging is preferable over macro-averaging

        :param Literal[&#39;micro&#39;, &#39;macro&#39;] agg: aggregation method
        :return pd.DataFrame: containing [threshold, precision, recall]
        """
        y_true_binarized = label_binarize(
            self.y_true, 
            classes = list(range(self.y_pred_probs.shape[1]))
        )
        if agg == 'micro':
            precision, recall, threshold = precision_recall_curve(
                y_true_binarized.ravel(), 
                self.y_pred_probs.ravel(),
            )
            threshold = np.array(list(threshold) + [1])

        elif agg == 'macro':
            precisions, recalls, thresholds = dict(), dict(), dict()
            for i in range(self.num_cls):
                precisions[i], recalls[i], thresholds[i] = precision_recall_curve(
                    y_true_binarized[:, i], 
                    self.y_pred_probs[:, i],
                )
                # precision and recall have n+1 elements while threshold only have n
                # precision[n+1] = 1, recall[n+1] = 0
                thresholds[i] = np.array(list(thresholds[i]) + [1])
                    
    
            threshold = np.linspace(0.0, 1.0, 2500)
            precision = np.zeros_like(threshold)
            recall = np.zeros_like(threshold)
            for i in range(self.num_cls):
                # linear interpolation
                precision += np.interp(threshold, thresholds[i], precisions[i]) 
                recall += np.interp(threshold, thresholds[i], recalls[i])
                
            # Average it and compute AUC
            precision /= self.num_cls
            recall /= self.num_cls
            
        else:
            raise ValueError("only support micro or macro aggregation")
        
        return pd.DataFrame({
                'threshold': threshold,
                'precision': precision,
                'recall': recall
            })
