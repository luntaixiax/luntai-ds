from typing import List, Literal, Tuple
import ibis
from ibis import _
import numpy as np
import pandas as pd
from luntaiDs.ModelingTools.Explore.engines.base import _BaseEDAEngine, _BaseNumericHelper, serialize
from luntaiDs.ModelingTools.Explore.summary import DescStat, QuantileStat, StatVar, XtremeStat, \
    BinaryStatAttr, BinaryStatSummary, CategStatAttr, \
    CategStatSummary, NumericStatAttr, NumericStatSummary
        

class NumericHelperIbis(_BaseNumericHelper):
    def __init__(self, df: ibis.expr.types.Table, colname: str):
        self._df = df
        self._colname = colname
    
    def get_descriptive_stat(self) -> DescStat:
            # get min/max
        descrp = (
            self._df
            .mutate(_[self._colname].cast('Float64').name(self._colname))
            .mutate(NORM_ = (_[self._colname] - _[self._colname].mean()))
            .mutate(
                NORM3_ = _['NORM_'] * _['NORM_'] * _['NORM_'],
                NORM4_ = _['NORM_'] * _['NORM_'] * _['NORM_'] * _['NORM_'],
                NORMABS_ = _['NORM_'].abs()
            )
            .aggregate(
                MIN_ = _[self._colname].min(),
                MAX_ = _[self._colname].max(),
                MEAN_ = _[self._colname].mean(),
                VAR_ = _[self._colname].var(),
                STD_ = _[self._colname].std(),
                MAD_ = _['NORMABS_'].mean(),
                NORM3_ = _['NORM3_'].mean(),
                NORM4_ = _['NORM4_'].mean()
            )
            .to_pandas()
            .astype('float')
        )
        
        mean_ = descrp.loc[0, 'MEAN_']
        var_ = descrp.loc[0, 'VAR_']
        std_ = descrp.loc[0, 'STD_']
        mad_ = descrp.loc[0, 'MAD_'] # avg(abs(X-u))
        norm3_ = descrp.loc[0, 'NORM3_'] # avg((X-u)^3)
        norm4_ = descrp.loc[0, 'NORM4_'] # avg((X-u)^4)
        
            
        # descriptive statistics
        stat_descriptive = DescStat(
            mean=serialize(mean_),
            var=serialize(var_),
            std=serialize(std_),
            skew=serialize(norm3_ / std_ ** 3), # TODO: whether subtract 3
            kurt=serialize(norm4_ / std_ ** 4),
            mad=serialize(mad_),
            cv=serialize(std_ / mean_),
            normality_p=None,
        )
        return stat_descriptive
    
    def getQuantiles(self, quantiles: List[float]) -> List[float]:
        kws = dict()
        for q in quantiles:
            kws[str(q)] = _[self._colname].quantile(q)
        qs = self._df.aggregate(**kws).to_pandas()
        return qs.iloc[0, :].astype('float').tolist()
    
    def get_quantile_stat(self) -> QuantileStat:
        descrp = (
            self._df
            .aggregate(
                MIN_ = _[self._colname].min(),
                MAX_ = _[self._colname].max(),
            )
            .to_pandas()
            .astype('float')
        )
        
        min_ = descrp.loc[0, 'MIN_']
        max_ = descrp.loc[0, 'MAX_']
        
        # get quantiles
        try:
            q001, q005, q025, q05, q075, q095, q099 = self.getQuantiles(
                self._df, 
                colname=self._colname, 
                quantiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            )
            
        except Exception as e:
            # use ntile as backup
            quantiles = (
                self._df
                .mutate(
                    NTILE_ = ibis.ntile(100).over(
                        order_by=_[self._colname]
                    )
                )
                .group_by('NTILE_')
                .aggregate(
                    MIN_ = _[self._colname].min(), # 0=0%, 1=1%, 2=2%
                    #MAX_ = _[colname].max(),
                )
                .to_pandas()
                .set_index('NTILE_')
                .astype('float')
            )
            q001 = quantiles.loc[1, 'MIN_']
            q005 = quantiles.loc[5, 'MIN_']
            q025 = quantiles.loc[25, 'MIN_']
            q05 = quantiles.loc[50, 'MIN_']
            q075 = quantiles.loc[75, 'MIN_']
            q095 = quantiles.loc[95, 'MIN_']
            q099 = quantiles.loc[99, 'MIN_']
            
            
        stat_quantile = QuantileStat(
            minimum=serialize(min_),
            perc_1th=serialize(q001),
            perc_5th=serialize(q005),
            q1=serialize(q025),
            median=serialize(q05),
            q3=serialize(q075),
            perc_95th=serialize(q095),
            perc_99th=serialize(q099),
            maximum=serialize(max_),
            iqr=serialize(q075 - q025),
            range=serialize(max_ - min_),
        )

        return stat_quantile        
        
    
    def get_xtreme_stat(self, xtreme_method: Literal["iqr", "quantile"] = "iqr") -> XtremeStat:
        if xtreme_method == "iqr":
            try:
                q1, q3 = self.getQuantiles(
                    self._df, 
                    colname=self._colname, 
                    quantiles=[0.25, 0.75]
                )
            except Exception as e:
                # use ntile as backup
                quantiles = (
                    self._df
                    .mutate(
                        NTILE_ = ibis.ntile(4).over(
                            order_by=_[self._colname]
                        )
                    )
                    .group_by('NTILE_')
                    .aggregate(
                        MIN_ = _[self._colname].min(), # 0=0%, 1=25%, 2=50%
                        #MAX_ = _[colname].max(),
                    )
                    .to_pandas()
                    .set_index('NTILE_')
                    .astype('float')
                )
                q1 = quantiles.loc[1, 'MIN_']
                q3 = quantiles.loc[3, 'MIN_']
            
            iqr = q3 - q1
            lbound = q1 - 1.5 * iqr
            rbound = q3 + 1.5 * iqr

        elif xtreme_method == "quantile":
            try:
                lbound, rbound = self.getQuantiles(
                    self._df, 
                    colname=self._colname, 
                    quantiles=[0.01, 0.99]
                )
                
            except Exception as e:
                # use ntile as backup
                quantiles = (
                    self._df
                    .mutate(
                        NTILE_ = ibis.ntile(100).over(
                            order_by=_[self._colname]
                        )
                    )
                    .group_by('NTILE_')
                    .aggregate(
                        MIN_ = _[self._colname].min(), # 0=0%, 1=1%, 2=2%
                        #MAX_ = _[self._colname].max(),
                    )
                    .to_pandas()
                    .set_index('NTILE_')
                    .astype('float')
                )
                lbound = quantiles.loc[1, 'MIN_']
                rbound = quantiles.loc[99, 'MIN_']
        else:
            raise ValueError("xtreme_method can only be iqr or quantile")

        xtremes = (
            self._df
            .aggregate(
                lxtreme_mean = _[self._colname].mean(
                    where = _[self._colname] < lbound
                ),
                lxtreme_median = _[self._colname].approx_median(
                    where = _[self._colname] < lbound
                ),
                rxtreme_mean = _[self._colname].mean(
                    where = _[self._colname] > rbound
                ),
                rxtreme_median = _[self._colname].approx_median(
                    where = _[self._colname] > rbound
                )
            )
            .to_pandas()
            .astype('float')
        )

        return XtremeStat(
            lbound=serialize(lbound),
            rbound=serialize(rbound),
            lxtreme_mean=serialize(xtremes.loc[0, 'lxtreme_mean']),
            lxtreme_median=serialize(xtremes.loc[0, 'lxtreme_median']),
            rxtreme_mean=serialize(xtremes.loc[0, 'rxtreme_mean']),
            rxtreme_median=serialize(xtremes.loc[0, 'rxtreme_median']),
        )
        
    def get_histogram(self, n_bins:int) -> Tuple[np.ndarray, np.ndarray]:
        # get bin edges
        stat = (
            self._df
            .aggregate(
                MIN_ = _[self._colname].min(),
                MAX_ = _[self._colname].max()
            )
            .to_pandas()
            .astype('float')
        )
        min_ = stat.loc[0, 'MIN_']
        max_ = stat.loc[0, 'MAX_']
        bin_edges_ = np.linspace(min_, max_, n_bins + 1)
        
        # get count of each bins
        hist = (
            self._df
            .mutate(
                _[self._colname]
                .histogram(nbins = n_bins)
                .name('HIST_')
            )
            .group_by('HIST_')
            .aggregate(
                # MIN_ = _[self._colname].min(),
                # MAX_ = _[self._colname].max(),
                NUM_ = _[self._colname].count()
            )
            .to_pandas()
        )
        # need to clip because the max value will be at edge
        hist['HIST_'] = hist['HIST_'].clip(0, n_bins - 1)
        
        hist_ = (
            hist
            .set_index('HIST_')
            .join(
                pd.DataFrame(index = np.arange(n_bins)),
                how = 'right'
            )
            .fillna(0)
            ['NUM_'].astype('int')
            .values
        )
        return hist_, bin_edges_


class EDAEngineIbis(_BaseEDAEngine):
    def __init__(self, df: ibis.expr.types.Table):
        self._df = df
    
    def _fit_common_categ(self, colname: str, attr: CategStatAttr) -> CategStatSummary:
        """common categorical variable fitting can be reused by subclass categorical fitting

        :param str colname: column name
        :param CategStatAttr attr: categorical variable attribute object
        :return CategStatSummary: categorical variable summary object
        """
        basic = (
            self._df
            .aggregate(
                total_ = _.count(),
                missing_ = _.count(where = _[colname].isnull()),
                unique_ = _[colname].nunique() # already dropped na
            )
            .to_pandas()
        )
        # total number of records
        total_ = serialize(basic.loc[0, 'total_'])
        # missing rate
        missing = basic.loc[0, 'missing_']
        missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # num of unique value (dropped NA value)
        unique = basic.loc[0, 'unique_']
        unique_ = StatVar(
            value=serialize(unique), 
            perc=serialize(unique / total_)
        )
        
        distribution = (
            self._df
            .dropna([colname])
            [colname]
            .value_counts()
            .to_pandas()
        )

        # category counts without null
        vcounts_ = pd.Series(
            distribution.iloc[:, 1].values,
            index = distribution.iloc[:, 0],
            name = colname
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
        # reuse common stat
        common_stat_summary = self._fit_common_categ(colname, attr)
        # binary stat
        binary = (
            ibis
            .case()
            .when(self._df[colname].isnull(), int(attr.na_to_pos_))
            .when(self._df[colname].isin(attr.pos_values_), 1)
            .else_(0)
            .end()
            .value_counts()
            .to_pandas()
        )
        binary_vcounts_ = pd.Series(
            binary.iloc[:, 1].values,
            index = binary.iloc[:, 0],
            name = colname
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
        df = self._df.mutate(_[colname].cast('Float64').name(colname))
        basic = (
            df
            .aggregate(
                total_ = _.count(),
                missing_ = _.count(where = (_[colname].isnull()) | (_[colname].isnan())),
                zeros_ =  _.count(where = _[colname] == 0),
                infs_pos_ = _.count(where = (_[colname] > 0) & (_[colname].isinf())),
                infs_neg_ = _.count(where = (_[colname] < 0) & (_[colname].isinf())),
            )
            .to_pandas()
        )
        # total number of records
        total_ = serialize(basic.loc[0, 'total_'])
        # missing rate
        missing = basic.loc[0, 'missing_']
        missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # zero rate
        zeros = basic.loc[0, 'zeros_']
        zeros_ = StatVar(
            value=serialize(zeros), 
            perc=serialize(zeros / total_)
        )
        # infinite value
        infs_pos = basic.loc[0, 'infs_pos_']
        infs_pos_ = StatVar(
            value=serialize(infs_pos), 
            perc=serialize(infs_pos / total_)
        )
        infs_neg = basic.loc[0, 'infs_neg_']
        infs_neg_ = StatVar(
            value=serialize(infs_neg), 
            perc=serialize(infs_neg / total_)
        )
        
        # clean version
        df = (
            df
            .dropna([colname])
            .filter(~_[colname].isinf())
        )
        if attr.setaside_zero_:
            df = df.filter(_[colname] != 0)
        n_exextreme = df.count().to_pandas()
        
        # xtreme value
        if attr.xtreme_method_:
            xstat = NumericHelperIbis(
                df = df,
                colname = colname
            )
            xtreme_stat_ = xstat.get_xtreme_stat(
                xtreme_method=attr.xtreme_method_
            )
            df_clean = df.filter(
                (_[colname] >= xtreme_stat_.lbound) 
                & (_[colname] <= xtreme_stat_.rbound)
            )
            num_xtreme = (n_exextreme - df_clean.count()).to_pandas()
        else:
            df_clean = df
            num_xtreme = 0
            xtreme_stat_ = None
        
        xtreme_ = StatVar(
            value=serialize(num_xtreme), 
            perc=serialize(num_xtreme / total_)
        )
        
        # percentile & descriptive statistics
        xstat_clean = NumericHelperIbis(
            df = df_clean,
            colname = colname
        )
        stat_quantile_ = xstat_clean.get_quantile_stat()
        stat_descriptive_ = xstat_clean.get_descriptive_stat()
        
        # histogram
        hist_, bin_edges_ = xstat_clean.get_histogram(
            n_bins = attr.bins_
        )
        
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
            colname_log = f"{colname}_log"
            df_log = df_clean.mutate(_[colname].log10().name(colname_log))
            
            # xtreme value
            if attr.xtreme_method_:
                xstat_log = NumericHelperIbis(
                    df = df_log,
                    colname = colname_log
                )
                num_stat.xtreme_stat_log_ = xstat_log.get_xtreme_stat(
                    xtreme_method=attr.xtreme_method_
                )
                df_clean_log = df_log.filter(
                    (_[colname_log] >= num_stat.xtreme_stat_log_.lbound) 
                    & (_[colname_log] <= num_stat.xtreme_stat_log_.rbound)
                )
                num_xtreme_log = (n_exextreme - df_clean_log.count()).to_pandas()
            else:
                df_clean_log = df_log
                num_xtreme_log = 0
                num_stat.xtreme_stat_log_ = None
            
            num_stat.xtreme_log_ = StatVar(
                value=serialize(num_xtreme_log), 
                perc=serialize(num_xtreme_log / total_)
            )
            
            # percentile & descriptive statistics
            xstat_clean_log = NumericHelperIbis(
                df = df_clean_log,
                colname = colname
            )
            num_stat.stat_quantile_log_ = xstat_clean_log.get_quantile_stat()
            num_stat.stat_descriptive_log_ = xstat_clean_log.get_descriptive_stat()

            # histogram
            num_stat.hist_log_, num_stat.bin_edges_log_ = xstat_clean_log.get_histogram(
                n_bins = attr.bins_
            )
            
        return num_stat