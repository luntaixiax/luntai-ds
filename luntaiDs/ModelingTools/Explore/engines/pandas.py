from typing import Literal
import numpy as np
from numpy import expm1, log1p, log10
import pandas as pd
from scipy.stats import rv_histogram, chi2_contingency, variation, kstest

from luntaiDs.ModelingTools.Explore.engines.base import _BaseEDAEngine, serialize
from luntaiDs.ModelingTools.Explore.profiling import DescStat, QuantileStat, StatVar, XtremeStat
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, BinaryStatSummary, CategStatAttr, \
    CategStatSummary, NominalCategStatAttr, NominalCategStatSummary, NumericStatAttr, \
        NumericStatSummary, OrdinalCategStatAttr, OrdinalCategStatSummary


    
    
def getNumericalStat(vector) -> tuple[QuantileStat, DescStat]:
    """Get the numerical statistics

    Args:
        vector: the underlying data

    Returns: [quantile statistics, descriptive statistics]

    """
    vector = vector.astype("float")
    stat_quantile = QuantileStat(
        minimum=serialize(vector.min()),
        perc_1th=serialize(vector.quantile(0.01)),
        perc_5th=serialize(vector.quantile(0.05)),
        q1=serialize(vector.quantile(0.25)),
        median=serialize(vector.quantile(0.5)),
        q3=serialize(vector.quantile(0.75)),
        perc_95th=serialize(vector.quantile(0.95)),
        perc_99th=serialize(vector.quantile(0.99)),
        maximum=serialize(vector.max()),
        iqr=serialize(vector.quantile(0.75) - vector.quantile(0.25)),
        range=serialize(vector.max() - vector.min()),
    )
    # descriptive statistics
    stat_descriptive = DescStat(
        mean=serialize(vector.mean()),
        var=serialize(vector.var()),
        std=serialize(vector.std()),
        skew=serialize(vector.skew()),
        kurt=serialize(vector.kurtosis()),
        mad=serialize((vector - vector.mean()).abs().mean()), # mean absolute deviation
        cv=serialize(variation(vector, nan_policy="omit")),
        normality_p=serialize(kstest(vector.dropna(), cdf="norm")[1]),
    )
    return stat_quantile, stat_descriptive


def getXtremeStat(vector, xtreme_method: Literal["iqr", "quantile"] = "iqr") -> XtremeStat:
    """Get extreme value statistics

    Args:
        vector: the underlying data
        xtreme_method: the extreme detection method, iqr or quantile

    Returns: the extreme statistics

    """
    vector = vector.astype("float")
    if xtreme_method == "iqr":
        q1 = vector.quantile(0.25)
        q3 = vector.quantile(0.75)
        iqr = q3 - q1
        lbound = q1 - 1.5 * iqr
        rbound = q3 + 1.5 * iqr

    elif xtreme_method == "quantile":
        lbound = vector.quantile(0.01)
        rbound = vector.quantile(0.99)
    else:
        raise ValueError("xtreme_method can only be iqr or quantile")

    lvector = vector[vector < lbound]
    rvector = vector[vector > rbound]

    return XtremeStat(
        lbound=serialize(lbound),
        rbound=serialize(rbound),
        lxtreme_mean=serialize(lvector.mean()),
        lxtreme_median=serialize(lvector.median()),
        rxtreme_mean=serialize(rvector.mean()),
        rxtreme_median=serialize(rvector.median()),
    )

def log10pc(x):
    """Do log10p transform on both positive and negative range

    Args:
        x: the original value

    Returns: value after log10p transform

    """
    return np.where(x > 0, log10(x + 1), -log10(1 - x))

class EDAEnginePandas(_BaseEDAEngine):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    def _fit_common_categ(self, colname: str, attr: CategStatAttr) -> CategStatSummary:
        """common categorical variable fitting can be reused by subclass categorical fitting

        :param str colname: column name
        :param CategStatAttr attr: categorical variable attribute object
        :return CategStatSummary: categorical variable summary object
        """
        vector = self._df[colname]
        
        # total number of records
        total_ = len(vector)
        # missing rate
        missing = vector.isna()
        missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # num of unique value (dropped NA value)
        unique = len(vector.dropna().unique())
        unique_ = StatVar(
            value=serialize(unique), 
            perc=serialize(unique / total_)
        )
        # category counts without null
        vcounts_ = (
            vector
            .value_counts(dropna=True, normalize=False)
            .fillna(0)
        )
        vpercs_ = (
            vcounts_ / vcounts_.sum()
        ).fillna(0)
        
        return CategStatSummary(
            colname_ = colname,
            total_ = total_,
            missing_ = missing_,
            unique_ = unique_,
            vcounts_ = vcounts_,
            vpercs_ = vpercs_
        )
        
    def fit_binary(self, colname: str, attr: BinaryStatAttr) -> BinaryStatSummary:
        """binary categorical variable fitting

        :param str colname: column name
        :param BinaryStatAttr attr: binary variable attribute object
        :return BinaryStatSummary: binary variable summary object
        """
        vector = self._df[colname]
        
        # reuse common stat
        common_stat_summary = self._fit_common_categ(colname, attr)
        #num_unique = common_stat_summary.unique_.value
        #assert num_unique <= 2, f"Labeled as binary value, but {num_unique} categories are found"
        
        # binary stat
        vector_binary = vector.copy()
        vector_binary[~vector_binary.isna()] = vector_binary[~vector_binary.isna()].apply(
            lambda v: 1 if v in attr.pos_values_ else 0
        )
        vector_binary = vector_binary.fillna(int(attr.na_to_pos_))
        binary_vcounts_ = (
            vector_binary
            .value_counts(dropna=True, normalize=False)
            .fillna(0)
        )
        binary_vpercs_ = (
            binary_vcounts_ / binary_vcounts_.sum()
        ).fillna(0)
        
        return BinaryStatSummary(
            colname_ = common_stat_summary.colname_,
            total_ = common_stat_summary.total_,
            missing_ = common_stat_summary.missing_,
            unique_ = common_stat_summary.unique_,
            vcounts_ = common_stat_summary.vcounts_,
            vpercs_ = common_stat_summary.vpercs_,
            binary_vcounts_ = binary_vcounts_,
            binary_vpercs_ = binary_vpercs_
        )
        
    def fit_numeric(self, colname: str, attr: NumericStatAttr) -> NumericStatSummary:
        """numeric variable fitting

        :param str colname: column name
        :param NumericStatAttr attr: numeric variable attribute object
        :return NumericStatSummary: numeric variable summary object
        """
        vector = self._df[colname].astype('float')
        
        # total number of records
        total_ = len(vector)
        # missing rate
        missing = vector.isna()
        missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # zero rate
        zeros = vector[vector == 0].count()
        zeros_ = StatVar(
            value=serialize(zeros), 
            perc=serialize(zeros / total_)
        )
        
        # infinite value
        infs_pos = vector[vector == np.inf].count()
        infs_pos_ = StatVar(
            value=serialize(infs_pos), 
            perc=serialize(infs_pos / total_)
        )
        infs_neg = vector[vector == -np.inf].count()
        infs_neg_ = StatVar(
            value=serialize(infs_neg), 
            perc=serialize(infs_neg / total_)
        )
        
        vector = vector.dropna()
        vector = vector[(vector < np.inf) & (vector > -np.inf)]
        if attr.setaside_zero_:
            vector = vector[vector != 0]
            
        # xtreme value
        if attr.xtreme_method_:
            xtreme_stat_ = getXtremeStat(vector, xtreme_method=attr.xtreme_method_)
            vector_clean = vector[
                (vector >= xtreme_stat_.lbound) & (vector <= xtreme_stat_.rbound)
            ]

        else:
            vector_clean = vector.copy()
            xtreme_stat_ = None
            
        num_xtreme = len(vector) - len(vector_clean)
        xtreme_ = StatVar(
            value=serialize(num_xtreme), 
            perc=serialize(num_xtreme / total_)
        )
        
        # percentile & descriptive statistics
        stat_quantile_, stat_descriptive_ = getNumericalStat(vector_clean)
        # histogram
        hist_, bin_edges_ = np.histogram(vector_clean, bins=attr.bins_)
        
        num_stat = NumericStatSummary(
            colname_ = colname,
            total_ = total_,
            missing_ = missing_,
            zeros_ = zeros_,
            infs_pos_ = infs_pos_,
            infs_neg_ = infs_neg_,
            stat_quantile_ = stat_quantile_,
            stat_descriptive_ = stat_descriptive_,
            hist_ = hist_,
            bin_edges_ = bin_edges_,
            xtreme_ = xtreme_,
            xtreme_stat_ = xtreme_stat_
        )
        
        if attr.log_scale_:
            vector_log = pd.Series(
                log10pc(vector), 
                name=colname, 
                dtype="float"
            )

            # xtreme value
            if attr.xtreme_method_:
                num_stat.xtreme_stat_log_ = getXtremeStat(vector_log, xtreme_method=attr.xtreme_method_)
                vector_clean_log = vector_log[
                    (vector_log >= num_stat.xtreme_stat_log_.lbound)
                    & (vector_log <= num_stat.xtreme_stat_log_.rbound)
                ]
            else:
                vector_clean_log = vector_log.copy()

            num_xtreme_log = len(vector_log) - len(vector_clean_log)
            num_stat.xtreme_log_ = StatVar(
                value=serialize(num_xtreme_log), 
                perc=serialize(num_xtreme_log / total_)
            )
            # percentile & descriptive statistics
            num_stat.stat_quantile_log_, num_stat.stat_descriptive_log_ = getNumericalStat(vector_clean_log)
            # histogram
            num_stat.hist_log_, num_stat.bin_edges_log_ = np.histogram(vector_clean_log, bins=attr.bins_)
            
        return num_stat