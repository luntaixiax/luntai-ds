from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import cycle
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, classification_report, auc, \
    roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import label_binarize


class Validator:
    pass

class ClfValidator(Validator):
    def __init__(self, clf, Xs, y_actual, y_labels=None, binary:bool = True):
        self.binary = binary
        self.clf = clf
        self.Xs = Xs
        self.y_actual = y_actual
        self.y_predict_cv = clf.predict(Xs)
        self.y_scores = clf.predict_proba(Xs)

        # map y labels
        if y_labels is None:
            self.y_labels = clf.classes_
        else:
            self.y_labels = y_labels

        self.y_actual_binary = label_binarize(self.y_actual, classes=self.y_labels)
        if len(self.y_labels) == 2 or binary:  # adjust for binary classifier -- increase to 2D dimension
            self.y_actual_binary = np.c_[1 - self.y_actual_binary, self.y_actual_binary]

    def confusion_matrix(self, plot: bool = True, figsize=(12, 5)):

        cm = confusion_matrix(self.y_actual, self.y_predict_cv)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float)
        np.fill_diagonal(cm_perc, 0)  # fill diagonal value for error chart to zero

        cm = pd.DataFrame(cm, index=self.y_labels, columns=self.y_labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        cm_perc = pd.DataFrame(cm_perc, index=self.y_labels, columns=self.y_labels)
        cm_perc.index.name = 'Actual'
        cm_perc.columns.name = 'Predicted'

        if plot:
            annot1 = np.empty_like(cm).astype(str)  # absolute value chart (colorful heatmap)
            annot2 = np.empty_like(cm).astype(str)  # error chart (gray scale)
            nrows, ncols = cm.shape
            for i in range(nrows):
                for j in range(ncols):
                    c = cm.iloc[i, j]
                    p = cm_perc.iloc[i, j]
                    if c == 0:
                        annot1[i, j] = ''
                        annot2[i, j] = ''
                    else:
                        annot1[i, j] = f'{c : .0f}'

                        if i != j:
                            annot2[i, j] = f'{p :.0%}'
                        else:
                            annot2[i, j] = ''

            fig, axes = plt.subplots(1, 2, sharex=True, figsize=figsize)
            sns.heatmap(ax=axes[0], data=cm, annot=annot1, fmt='')
            sns.heatmap(ax=axes[1], data=cm_perc, annot=annot2, fmt='', cmap=plt.cm.gray, vmin=0, vmax=1)

            axes[0].set_title('Confusion Matrix')
            axes[1].set_title('Error Heatmap - p(predict | actual)')
        return cm, cm_perc

    def average_precision_score(self, average: str = "macro") -> float:
        """

        :param average: {‘micro’, ‘macro’}, default='macro'
        """
        return average_precision_score(self.y_actual_binary, self.y_scores, average = average)

    def precision_recall(self, average: str = "macro", plot: bool = True, figsize = (5, 5)) -> dict:
        """

        :param average: {‘micro’, ‘macro’}, default='macro'
        :param plot: whether to plot or not
        :return: a dictionary {label : df[threshold, precision, recall]}
        """
        results = dict()
        aps = dict()

        for i, label in enumerate(self.y_labels):
            precision, recall, threshold = precision_recall_curve(self.y_actual_binary[:, i], self.y_scores[:, i])
            results[label] = pd.DataFrame({
                'threshold': threshold,
                'precision': precision[:-1],
                'recall': recall[:-1]
            })
            aps[label] = auc(recall, precision)

        # average Precision, Recall (micro/macro)
        if average == 'micro':
            ## micro ROC
            pre_avg, rec_avg, threshold_avg = precision_recall_curve(self.y_actual_binary.ravel(), self.y_scores.ravel())
        elif average == 'macro':
            ## macro ROC
            all_prec = np.unique(np.concatenate([results[label]['precision'] for label in self.y_labels]))

            #### Then interpolate all ROC curves at this points
            mean_rec = np.zeros_like(all_prec)
            for label in self.y_labels:
                mean_rec += np.interp(all_prec, results[label]['precision'], results[label]['recall'])

            mean_rec /= len(self.y_labels)

            pre_avg = all_prec
            rec_avg = mean_rec
            threshold_avg = np.empty(pre_avg.shape[0] - 1)
            threshold_avg[:] = np.nan
        else:
            raise ValueError("parameter 'average' can only be micro or macro")

        results['AVG'] = pd.DataFrame({
            'threshold': threshold_avg,
            'precision': pre_avg[:-1],
            'recall': rec_avg[:-1]
        })

        precision_avg = self.average_precision_score(average=average)
        precision_overall = self.average_precision_score(average='weighted')

        if plot:
            sns.set_style("ticks", {'axes.grid': True})
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            for i, label in enumerate(self.y_labels):
                legend = f"{label} (AP = {aps[label]:.3f})"
                sns.lineplot(x = results[label]['recall'], y = results[label]['precision'], label = legend, linewidth=2)

            # average Precision:
            legend = f"{average} average AP = {precision_avg:.3f}"
            sns.lineplot(x=rec_avg, y=pre_avg, label=legend, linewidth=4, color="navy", linestyle="--")

            axes.set_xlabel("recall")
            axes.set_ylabel("precision")
            axes.set_title(f"precision vs. recall curve (Weighted AP = {precision_overall:.3f})")
            axes.set_xlim([0, 1])
            axes.set_ylim([0, 1])

            plt.legend(loc="best")
        return results

    def roc_auc_score(self, multi_class: str = "ovo", average: str = "macro") -> float:
        """

        :param multi_class: {‘ovr’, ‘ovo’}, default='ovo'
        :param average: {‘micro’, ‘macro’}, default='macro'
        """
        return roc_auc_score(self.y_actual_binary, self.y_scores, multi_class = multi_class, average = average)


    def roc_curve(self, multi_class: str = "ovo", average: str = "macro", plot: bool = True, figsize=(5, 5)) -> dict:
        """

        :param multi_class: {‘ovr’, ‘ovo’}, default='ovo'
        :param average: {‘micro’, ‘macro’}, default='macro'
        :param plot: whether to plot or not
        :param figsize:
        :return: a dictionary {label : df[threshold, fpr, tpr]}
        """
        if self.binary:
            average = 'macro'  # binary only support macro, or will be mistakenly wrong

        results = dict()
        aucs = dict()

        # fpr, tpr and AUC by class
        for i, label in enumerate(self.y_labels):
            fpr, tpr, threshold = roc_curve(self.y_actual_binary[:, i], self.y_scores[:, i], drop_intermediate=False)
            results[label] = pd.DataFrame({
                'threshold': threshold,
                'fpr': fpr,
                'tpr': tpr
            })
            aucs[label] = auc(fpr, tpr)

        # average ROC, AUC (micro/macro)
        if average == 'micro':
            ## micro ROC
            fpr_avg, tpr_avg, threshold_avg = roc_curve(self.y_actual_binary.ravel(), self.y_scores.ravel(),
                                                        drop_intermediate=False)
        elif average == 'macro':
            ## macro ROC
            all_fpr = np.unique(np.concatenate([results[label]['fpr'] for label in self.y_labels]))

            #### Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for label in self.y_labels:
                mean_tpr += np.interp(all_fpr, results[label]['fpr'], results[label]['tpr'])

            mean_tpr /= len(self.y_labels)

            fpr_avg = all_fpr
            tpr_avg = mean_tpr
            threshold_avg = np.empty(fpr_avg.shape)
            threshold_avg[:] = np.nan

        else:
            raise ValueError("parameter 'average' can only be micro or macro or binary")

        results['AVG'] = pd.DataFrame({
            'threshold': threshold_avg,
            'fpr': fpr_avg,
            'tpr': tpr_avg
        })

        roc_auc_avg = self.roc_auc_score( multi_class=multi_class, average=average)
        roc_auc_overall = self.roc_auc_score(multi_class=multi_class, average='weighted')  # weighted AUC

        if plot:
            sns.set_style("ticks", {'axes.grid': True})
            fig, axes = plt.subplots(1, 1, figsize=figsize)

            # ROC by class
            for i, label in enumerate(self.y_labels):
                legend = f"{label} (AUC = {aucs[label]:.3f})"
                sns.lineplot(x=results[label]['fpr'], y=results[label]['tpr'], label=legend, linewidth=2)

            # average ROC:
            legend = f"{average} AUC = {roc_auc_avg:.3f}"
            sns.lineplot(x=fpr_avg, y=tpr_avg, label=legend, linewidth=4, color="navy", linestyle="--")

            axes.set_xlabel("fpr")
            axes.set_ylabel("tpr")
            axes.set_title(f"ROC curve (Weighted AUC = {roc_auc_overall:.3f})")
            axes.set_xlim([0, 1])
            axes.set_ylim([0, 1])

            plt.legend(loc="best")
        return results


    def precision_recall_roc_by_class(self, class_, figsize=(12, 5)):
        """

        :param class_: class name string
        :return:
        """
        precision_recall = self.precision_recall(plot=False).get(class_)
        fpr_tpr = self.roc_curve(plot=False).get(class_)

        if precision_recall is None or fpr_tpr is None:
            raise ValueError(f"class_ {class_} not found in target variable, current target list = {self.y_labels}")

        sns.set_style("ticks", {'axes.grid': True})
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.lineplot(ax=axes[0], x=precision_recall['threshold'], y=precision_recall['precision'], label='Precision',
                     linewidth=2)
        sns.lineplot(ax=axes[0], x=precision_recall['threshold'], y=precision_recall['recall'], label='Recall',
                     linewidth=2)

        sns.lineplot(ax=axes[1], x=fpr_tpr['threshold'], y=fpr_tpr['tpr'], label='sensitivity(tpr)', linewidth=2)
        sns.lineplot(ax=axes[1], x=fpr_tpr['threshold'], y=1 - fpr_tpr['fpr'], label='specificity(1-fpr)', linewidth=2)

        axes[0].set_xlabel("threshold")
        axes[0].set_title("Precision/Recall vs. Threshold")
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

        axes[1].set_xlabel("threshold")
        axes[1].set_title("Sensitivity/Specificity vs. Threshold")
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])

        plt.legend(loc="best")

    def report(self, output_dict=False, sample_weight=None):
        """

        :param output_dict: if True, will return report in dict format, False will only print
        :param sample_weight: array-like of shape (n_samples,)
        :return:
        """
        return classification_report(self.y_actual, self.y_predict_cv, target_names=self.y_labels,
                                     output_dict=output_dict, sample_weight=sample_weight)

class OrdinalClfValidator(Validator):
    def __init__(self, clf, Xs, y_actual, ordered_classes):
        self.clf = clf
        self.Xs = Xs
        self.y_actual = y_actual
        self.y_predict = clf.predict(Xs)
        self.y_labels = ordered_classes

        # label encode
        self.label_mappings = dict(zip(range(len(ordered_classes)), ordered_classes))
        self.reverse_label_mappings = dict(zip(ordered_classes, range(len(ordered_classes))))
        self.y_actual_label = pd.Series(self.y_actual).map(self.reverse_label_mappings).values
        self.y_predict_label = pd.Series(self.y_predict).map(self.reverse_label_mappings).values

    def conf_mat(self, error: bool = False) -> np.ndarray:

        cm = confusion_matrix(self.y_actual, self.y_predict)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float)
        np.fill_diagonal(cm_perc, 0)  # fill diagonal value for error chart to zero

        return cm_perc if error else cm

    def rank_matching(self, rate: bool = False, plot: bool = True) -> pd.DataFrame:
        # similar to 30/60/90: accuracy for: exact match/within 1 notch/within 2 notch,etc.
        n_ = len(self.y_actual_label)
        k_ = len(self.y_labels)

        # calculate diffs
        diff = self.y_predict_label - self.y_actual_label
        abs_diff = np.abs(diff)
        # calculate absolute matching numbers
        ## ensure the number of index is the same as total possible notches, i.e., still work when small errors
        matching = pd.Series(index=np.arange(-k_ + 1, k_), dtype='int')
        matching.update(pd.Series(diff).value_counts().sort_index())
        abs_matching = pd.Series(index=np.arange(k_), dtype='int')
        abs_matching.update(pd.Series(abs_diff).value_counts().sort_index())
        abs_matching.index = [f"±{idx}" for idx in range(k_)]
        abs_matching_cum = abs_matching.cumsum()
        abs_matching_cum.index = ["Exact"] + [f"w/i ±{idx}" for idx in range(1, k_)]
        # calculate absolute matching numbers by class
        ## ensure the number of index is the same as total possible notches, i.e., still work when small errorsx
        abs_matching_matrix = pd.DataFrame(index=range(k_), columns=self.y_labels)
        abs_matching_matrix.index.name = 'abs_diff'
        abs_matching_matrix.columns.name = 'actual'
        amnbc = pd.DataFrame({'actual': self.y_actual, 'abs_diff': abs_diff}).reset_index()
        abs_matching_matrix.update(pd.pivot_table(amnbc, values='index', index='abs_diff',
                                                  columns='actual', aggfunc='count').fillna(0).astype(
            'int').sort_index())
        abs_matching_matrix.index = [f"±{idx}" for idx in range(len(abs_matching.index))]
        abs_matching_matrix = abs_matching_matrix.fillna(0).astype('int')
        abs_matching_cum_matrix = abs_matching_matrix.cumsum(axis=0)
        abs_matching_cum_matrix.index = ["Exact"] + [f"w/i ±{idx}" for idx in range(1, k_)]

        # calculate matching rates
        matching_rt = matching / n_
        abs_matching_rt = abs_matching / n_
        abs_matching_cum_rt = abs_matching_cum / n_
        abs_matching_cum_rt_matrix = abs_matching_cum_matrix / np.repeat(
            np.sum(abs_matching_matrix.values, axis=0, keepdims=True), repeats=k_, axis=0)

        if plot:
            fig, axes = plt.subplot_mosaic(
                [['notch_rt', 'notch_rt', 'notch_grid1'],
                 ['abs_notch_rt', 'abs_notch_rt', 'notch_grid2']],
                figsize=(15, 10), constrained_layout=True)

            # 1. plot diff by notches
            sat = (np.abs(matching_rt.index) + 1) / (matching_rt.index.max() + 1)
            sns.barplot(x=matching_rt.index, y=matching_rt.values, ax=axes['notch_rt'], hue=np.abs(matching_rt.index),
                        palette="flare", dodge=False)
            for i in axes['notch_rt'].containers:
                axes['notch_rt'].bar_label(i, )
            axes['notch_rt'].set_title("Errors by notch")
            axes['notch_rt'].set_ylim([0, 1])

            # 2. plot abs_diff by class and as a whole piece
            colors = cycle(sns.color_palette("Set2"))
            for col in abs_matching_cum_rt_matrix.columns:
                sns.lineplot(x=abs_matching_cum_rt_matrix.index, y=abs_matching_cum_rt_matrix[col],
                             ax=axes['abs_notch_rt'], lw=2,
                             linestyle="--", label=col, color=next(colors))
            l_all = sns.lineplot(x=abs_matching_cum_rt.index, y=abs_matching_cum_rt.values, ax=axes['abs_notch_rt'],
                                 color="#24201E",
                                 marker='o', markersize=10, lw=5, label='Total')
            for idx, rt in abs_matching_cum_rt.iteritems():
                # l_all.text(idx, rt + 0.05, f"{rt:.0%}", horizontalalignment = 'center')
                l_all.annotate(f"{rt:.0%}", xy=(idx, rt), xytext=(idx, rt), textcoords='offset points',
                               bbox=dict(boxstyle="round4,pad=.5", fc="0.9"), horizontalalignment='center')
            axes['abs_notch_rt'].set_title("Matching rate by abs notch")
            axes['abs_notch_rt'].set_ylim([0, 1.1])

            # 3. plot heatmap for abs diff grid
            sns.heatmap(data=abs_matching_cum_matrix, annot=True, fmt="d", ax=axes['notch_grid1'])
            axes['notch_grid1'].set_title("Matching grid by class")

            annots = abs_matching_cum_rt_matrix.applymap(lambda x: f"{x:.0%}").values
            sns.heatmap(data=abs_matching_cum_rt_matrix, annot=annots, fmt="", cmap=plt.cm.gray, ax=axes['notch_grid2'])
            axes['notch_grid2'].set_title("Matching rate grid by class")

        return abs_matching_cum_rt_matrix if rate else abs_matching_cum_matrix

    def kendall_tau(self, alpha: float = 0.95) -> tuple:
        k, p = stats.kendalltau(self.y_actual, self.y_predict, nan_policy='omit')
        sig_flag = (p < (1 - alpha) / 2)
        return k, p, sig_flag

class ClfValidatorCV(ClfValidator):

    def __init__(self, clf, Xs, y_actual, y_labels=None, cv=3, binary:bool = False):
        self.binary = binary
        self.clf = clf
        self.Xs = Xs
        self.y_actual = y_actual
        self.y_predict_cv = cross_val_predict(clf, Xs, y_actual, cv=cv)
        self.y_scores = cross_val_predict(clf, Xs, y_actual, cv=cv, method="predict_proba")

        self.cv = cv
        # map y labels
        if y_labels is None:
            self.y_labels = clf.classes_
        else:
            self.y_labels = y_labels

        self.y_actual_binary = label_binarize(self.y_actual, classes=self.y_labels)
        if len(self.y_labels) == 2 or binary:  # adjust for binary classifier -- increase to 2D dimension
            self.y_actual_binary = np.c_[1 - self.y_actual_binary, self.y_actual_binary]



class ModelSelector:
    pass

class ClfModelSelector(ModelSelector):
    def __init__(self, clf_dict: dict, Xs, y_actual, y_labels = None, binary:bool = False):
        self.binary = binary
        self.models = {k : ClfValidator(clf, Xs, y_actual, y_labels, binary = binary) for k, clf in clf_dict.items()}

    def getClfv(self, model_name: str):
        return self.models.get(model_name)

    def confusion_matrix(self, plot: bool = True, figsize = (12, 5)) -> dict:
        conf_mats = {model_name : clf.confusion_matrix(plot = False) for model_name, clf in self.models.items()}

        n_models = len(self.models)
        figsize = (figsize[0], figsize[1] * n_models)
        if plot:
            fig, axes = plt.subplots(n_models, 2, sharex=True, figsize=figsize)

            for idx, (model_name, (cm, cm_perc)) in enumerate(conf_mats.items()):
                # annotation
                annot1 = np.empty_like(cm).astype(str)  # absolute value chart (colorful heatmap)
                annot2 = np.empty_like(cm).astype(str)  # error chart (gray scale)
                nrows, ncols = cm.shape
                for i in range(nrows):
                    for j in range(ncols):
                        c = cm.iloc[i, j]
                        p = cm_perc.iloc[i, j]
                        if c == 0:
                            annot1[i, j] = ''
                            annot2[i, j] = ''
                        else:
                            annot1[i, j] = f'{c : .0f}'

                            if i != j:
                                annot2[i, j] = f'{p :.0%}'
                            else:
                                annot2[i, j] = ''

                sns.heatmap(ax=axes[idx, 0], data = cm, annot = annot1, fmt = '')
                sns.heatmap(ax=axes[idx, 1], data = cm_perc, annot = annot2, fmt='', cmap=plt.cm.gray, vmin=0, vmax=1)

                axes[idx, 0].set_title(f'Confusion Matrix | {model_name}')
                axes[idx, 1].set_title(f'Error Heatmap - p(predict | actual) | {model_name}')

        return conf_mats


    def roc_curve(self, multi_class: str = "ovo", average: str = "macro", plot: bool = True, figsize=(5, 5)) -> dict:
        if self.binary:
            average = 'macro'

        avg_rocs = {}
        avg_aucs = {}
        for model_name, clfv in self.models.items():
            avg_rocs[model_name] = clfv.roc_curve(multi_class = multi_class, average = average, plot = False).get('AVG')
            avg_aucs[model_name] = clfv.roc_auc_score(multi_class = multi_class, average = average)

        if plot:
            sns.set_style("ticks", {'axes.grid': True})
            fig, axes = plt.subplots(1, 1, figsize=figsize)

            # ROC by model
            for model_name, avg_roc in avg_rocs.items():
                legend = f"{model_name} ({average} AUC = {avg_aucs[model_name]:.3f})"
                sns.lineplot(x = avg_roc['fpr'], y = avg_roc['tpr'], label = legend, linewidth = 2)

            axes.set_xlabel("fpr")
            axes.set_ylabel("tpr")
            axes.set_title(f"ROC curve")
            axes.set_xlim([0, 1])
            axes.set_ylim([0, 1])
            plt.legend(loc="best")

        return avg_rocs

    def precision_recall(self, average: str = "macro", plot: bool = True, figsize = (5, 5)) -> dict:
        avg_pre_recs = {}
        avg_aps = {}
        for model_name, clfv in self.models.items():
            avg_pre_recs[model_name] = clfv.precision_recall(average = average, plot = False).get('AVG')
            avg_aps[model_name] = clfv.average_precision_score(average = average)


        if plot:
            sns.set_style("ticks", {'axes.grid': True})
            fig, axes = plt.subplots(1, 1, figsize=figsize)

            # ROC by model
            for model_name, avg_pre_rec in avg_pre_recs.items():
                legend = f"{model_name} ({average} AP = {avg_aps[model_name]:.3f})"
                sns.lineplot(x = avg_pre_rec['precision'], y = avg_pre_rec['recall'], label = legend, linewidth = 2)

            axes.set_xlabel("precision")
            axes.set_ylabel("recall")
            axes.set_title(f"precision vs. recall curve")
            axes.set_xlim([0, 1])
            axes.set_ylim([0, 1])
            plt.legend(loc="best")

        return avg_pre_recs

class ClfModelSelectorCV(ClfModelSelector):
    def __init__(self, clf_dict: dict, Xs, y_actual, y_labels = None, binary:bool = False):
        self.binary = self.binary
        self.models = {k : ClfValidatorCV(clf, Xs, y_actual, y_labels, binary = binary) for k, clf in clf_dict.items()}

class OrdinalClfModelSelector(ModelSelector):
    def __init__(self, clf_dict: dict, Xs, y_actual, ordered_classes):
        self.models = {k : OrdinalClfValidator(clf, Xs, y_actual, ordered_classes) for k, clf in clf_dict.items()}

    def rank_matching(self, plot: bool = True, by_class: str = None) -> pd.DataFrame:
        matching_mats = pd.DataFrame(columns = ['model_name', 'band', 'matching'])
        for model_name, oclfv in self.models.items():
            if by_class:
                s = oclfv.rank_matching(rate = True, plot = False)[by_class]
                matching_mats = matching_mats.append(pd.DataFrame({'model_name' : model_name, 'band' : s.index, 'matching' : s.values}), ignore_index = True)
            else:
                s = oclfv.rank_matching(rate = False, plot = False).sum(axis = 1)
                s = s / s[-1]
                matching_mats = matching_mats.append(pd.DataFrame({'model_name' : model_name, 'band' : s.index, 'matching' : s.values}), ignore_index = True)

        if plot:
            fig, axes = plt.subplots(1, 1, figsize = (8, 8))

            sns.lineplot(data = matching_mats, x = 'band', y = 'matching', hue = 'model_name', lw = 3, marker="o", markersize = 10, dashes=False)
            if by_class:
                axes.set_title(f'Matching rate by notch - class {by_class}')
            else:
                axes.set_title('Matching rate by notch')

        return matching_mats

    def kendall_tau(self, alpha: float = 0.95, plot: bool = True) -> pd.DataFrame:
        kts = []
        for model_name, oclfv in self.models.items():
            kt, p, sig_flag = oclfv.kendall_tau(alpha = alpha)
            kts.append({'model_name' : model_name, 'kt' : kt, 'p' : p, 'sig' : sig_flag})

        kts = pd.DataFrame.from_records(kts)

        if plot:
            fig, axes = plt.subplots(1, 1, figsize = (6, 6))

            sns.barplot(data = kts, x = 'model_name', y = 'kt', palette = 'Set2')
            for idx, r in kts.iterrows():
                kt = r['kt']
                sig = r['sig']
                fc = '#90EC48' if sig else '#F5765A'  #if significant the facecolor of label will be green, otherwise red
                axes.annotate(f"{kt:.1%}", xy=(idx, kt), xytext=(idx, kt), textcoords='offset points',
                    bbox = dict(boxstyle="round4,pad=.5", fc = fc, alpha=0.8), horizontalalignment = 'center')

            axes.set_ylim([0, 1])
            axes.set_title('Kendall Tau correlation')

        return kts


class OrdinalClfModelSelectorCV(ClfModelSelectorCV):
    def __init__(self, clf_dict: dict, Xs, y_actual, y_labels = None):
        self.models = {k : ClfModelSelectorCV(clf, Xs, y_actual, y_labels) for k, clf in clf_dict.items()}