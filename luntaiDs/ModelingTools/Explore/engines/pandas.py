from typing import List, Literal, Tuple
import numpy as np
from optbinning import MulticlassOptimalBinning
import pandas as pd
from scipy.stats import variation, kstest
from luntaiDs.ModelingTools.Explore.engines.base import _BaseEDAEngine, _BaseNumericHelper, serialize
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, BinaryStatSummary, CategStatAttr, \
    CategStatSummary, CategUniVarClfTargetCorr, DescStat, NumericUniVarClfTargetCorr, QuantileStat, StatVar, XtremeStat, NumericStatAttr, \
        NumericStatSummary, exp10pc, log10pc


class NumericHelperPd(_BaseNumericHelper):
    def __init__(self, vector: pd.Series):
        self._vector = vector.astype("float")
    
    def get_descriptive_stat(self) -> DescStat:
        stat_descriptive = DescStat(
            mean=serialize(self._vector.mean()),
            var=serialize(self._vector.var()),
            std=serialize(self._vector.std()),
            skew=serialize(self._vector.skew()),
            kurt=serialize(self._vector.kurtosis()),
            mad=serialize((self._vector - self._vector.mean()).abs().mean()), # mean absolute deviation
            cv=serialize(variation(self._vector, nan_policy="omit")),
            normality_p=serialize(kstest(self._vector.dropna(), cdf="norm")[1]),
        )
        return stat_descriptive
    
    def get_quantile_stat(self) -> QuantileStat:
        stat_quantile = QuantileStat(
            minimum=serialize(self._vector.min()),
            perc_1th=serialize(self._vector.quantile(0.01)),
            perc_5th=serialize(self._vector.quantile(0.05)),
            q1=serialize(self._vector.quantile(0.25)),
            median=serialize(self._vector.quantile(0.5)),
            q3=serialize(self._vector.quantile(0.75)),
            perc_95th=serialize(self._vector.quantile(0.95)),
            perc_99th=serialize(self._vector.quantile(0.99)),
            maximum=serialize(self._vector.max()),
            iqr=serialize(self._vector.quantile(0.75) - self._vector.quantile(0.25)),
            range=serialize(self._vector.max() - self._vector.min()),
        )
        return stat_quantile        
        
    
    def get_xtreme_stat(self, xtreme_method: Literal["iqr", "quantile"] = "iqr") -> XtremeStat:
        if xtreme_method == "iqr":
            q1 = self._vector.quantile(0.25)
            q3 = self._vector.quantile(0.75)
            iqr = q3 - q1
            lbound = q1 - 1.5 * iqr
            rbound = q3 + 1.5 * iqr

        elif xtreme_method == "quantile":
            lbound = self._vector.quantile(0.01)
            rbound = self._vector.quantile(0.99)
        else:
            raise ValueError("xtreme_method can only be iqr or quantile")

        lvector = self._vector[self._vector < lbound]
        rvector = self._vector[self._vector > rbound]

        return XtremeStat(
            lbound=serialize(lbound),
            rbound=serialize(rbound),
            lxtreme_mean=serialize(lvector.mean()),
            lxtreme_median=serialize(lvector.median()),
            rxtreme_mean=serialize(rvector.mean()),
            rxtreme_median=serialize(rvector.median()),
        )
        
    def get_histogram(self, n_bins:int) -> Tuple[np.ndarray, np.ndarray]:
        hist_, bin_edges_ = np.histogram(self._vector, bins=n_bins)
        return hist_, bin_edges_

def _combine_x_y(x: pd.Series, y: pd.Series, 
        combine_x_categ:bool = False) -> pd.DataFrame:
    df = pd.DataFrame({'x' : x.values, 'y' : y.values})
    df = df.dropna(
        subset = ['y'], 
        axis = 0, 
        how = 'any'
    )
    # categorical variable for x only:
    if combine_x_categ:
        vc = df['x'].value_counts(normalize = True)
        others = vc.iloc[20:].index  # only keep top 20 categories
        df.loc[df['x'].isin(others), 'x'] = 'Others'
    
    return df


class EDAEnginePandas(_BaseEDAEngine):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    def get_columns(self) -> List[str]:
        """get all column list from given dataset

        :return List[str]: list of columns in the dataset
        """
        return self._df.columns.tolist()
        
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
            xst = NumericHelperPd(vector = vector)
            xtreme_stat_ = xst.get_xtreme_stat(
                xtreme_method=attr.xtreme_method_
            )
            vector_clean = vector[
                (vector >= xtreme_stat_.lbound) 
                & (vector <= xtreme_stat_.rbound)
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
        xst_clean = NumericHelperPd(vector = vector_clean)
        stat_quantile_ = xst_clean.get_quantile_stat()
        stat_descriptive_ = xst_clean.get_descriptive_stat()
        
        # histogram
        hist_, bin_edges_ = xst_clean.get_histogram(n_bins=attr.bins_)
        
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
                xst_log = NumericHelperPd(vector = vector_log)
                num_stat.xtreme_stat_log_ = xst_log.get_xtreme_stat(xtreme_method=attr.xtreme_method_)
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
            xstat_clean_log = NumericHelperPd(vector = vector_clean_log)
            
            num_stat.stat_quantile_log_ = xstat_clean_log.get_quantile_stat()
            num_stat.stat_descriptive_log_ = xstat_clean_log.get_descriptive_stat()
            # histogram
            num_stat.hist_log_, num_stat.bin_edges_log_ = xstat_clean_log.get_histogram(n_bins=attr.bins_)
            
        return num_stat
    
    def fit_univarclf_categ(self, x_col: str, y_col: str) -> CategUniVarClfTargetCorr:
        """fit univariate correlation for classififer target and categorical variable

        :param str x_col: column of a categorical variable
        :param str y_col: column of the classifier target varaible
        :return CategUniVarClfTargetCorr: result object
        """
        
        df = _combine_x_y(
            self._df[x_col].astype('str').fillna('Missing_'), # need to fill as separate categ
            self._df[y_col].astype('str'), 
            combine_x_categ = True # combine x's categories to be less than 20
        )
        
        # categorical:  p(x | y)
        p_x_y_ = {}
        for y in df['y'].unique():
            cs = (
                df
                .loc[df['y'] == y, ['x']]
                .groupby('x')
                .size()
            )
            d = pd.DataFrame({
                    'count' : cs.values, 
                    'perc' : cs  / cs.sum()
                }, 
                index = cs.index
            )
            p_x_y_[y] = d.fillna(0) #.to_dict(orient = 'index')
            
        # categorical:  p(y | x)  prob (event rate when binary clf) by category
        p_y_x_ = {}
        for x in df['x'].unique():
            cs = (
                df
                .loc[df['x'] == x, ['y']]
                .groupby('y')
                .size()
            )
            d = pd.DataFrame({
                    'count' : cs.values, 
                    'perc' : cs  / cs.sum()
                }, 
                index = cs.index
            )
            p_y_x_[x] = d.fillna(0) #.to_dict(orient = 'index')
            
        return CategUniVarClfTargetCorr(
            colname_ = x_col,
            yname_ = y_col,
            ylabels_ = (
                self._df[y_col]
                .dropna()
                .unique()
                .astype('str')
                .tolist()
            ),
            p_x_y_ = p_x_y_,
            p_y_x_ = p_y_x_
        )
    
    def fit_univarclf_numeric(self, x_col: str, y_col: str) -> NumericUniVarClfTargetCorr:
        """fit univariate correlation for classififer target and numeric variable

        :param str x_col: column of a numeric variable
        :param str y_col: column of the classifier target varaible
        :return NumericUniVarClfTargetCorr: result object
        """
        df = _combine_x_y(
            self._df[x_col].astype('float'), 
            self._df[y_col].astype('str'), 
        )
        
        # numerical:  p(x | y)  distribution by target (boxplot)
        t = {'origin' : {}, 'log' : {}}
        for y in df['y'].unique():
            # EDA on given subset
            eda = EDAEnginePandas(df[df['y'] == y])
            summary = eda.fit_numeric(
                colname = 'x', # already renamed
                attr = NumericStatAttr(
                    setaside_zero_ = False, 
                    log_scale_ = True, 
                    xtreme_method_ = 'iqr', 
                    bins_ = 100
                )
            )
            
            t['origin'][y] = {
                'lbound': summary.xtreme_stat_.lbound,
                'q1' : summary.stat_quantile_.q1,
                'mean' : summary.stat_descriptive_.mean,
                'median' : summary.stat_quantile_.median,
                'q3' : summary.stat_quantile_.q3,
                'rbound': summary.xtreme_stat_.rbound,
            }
            t['log'][y] = {
                'lbound': summary.xtreme_stat_log_.lbound,
                'q1' : summary.stat_quantile_log_.q1,
                'mean' : summary.stat_descriptive_log_.mean,
                'median' : summary.stat_quantile_log_.median,
                'q3' : summary.stat_quantile_log_.q3,
                'rbound': summary.xtreme_stat_log_.rbound,
            }
            
        p_x_y_ = {
            'origin' : pd.DataFrame.from_dict(t['origin'], orient='index'),
            'log' : pd.DataFrame.from_dict(t['log'], orient='index')
        }
            
            
        # numerical:  p(y | x)  prob (event rate if binary clf) by bucketized x
        mob = MulticlassOptimalBinning(
            prebinning_method = 'quantile',
            monotonic_trend = 'auto',
            max_n_prebins = 20,
            min_prebin_size = 0.05,
        )
        mob.fit(x = df['x'], y = df['y'])
        p_y_x_ = mob.binning_table.build(add_totals=False)
        
        return NumericUniVarClfTargetCorr(
            colname_ = x_col,
            yname_ = y_col,
            ylabels_ = (
                self._df[y_col]
                .dropna()
                .unique()
                .astype('str')
                .tolist()
            ),
            p_x_y_ = p_x_y_,
            p_y_x_ = p_y_x_
        )

    
