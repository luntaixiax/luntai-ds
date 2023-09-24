import pandas as pd
import numpy as np
from typing import Union, Dict, List, Literal
from collections import namedtuple
from scipy.special import exp10
from numpy import expm1, log1p, log10
from scipy.stats import rv_histogram, chi2_contingency, variation, kstest
from optbinning import MulticlassOptimalBinning

from ModelingTools.utils.parallel import parallel_run, delayer
from CommonTools.accessor import toJSON, loadJSON

# named tuple for value and percentage
StatVar = namedtuple(
    'StatVar', 
    field_names=['value', 'perc'], 
    defaults = [0, 0.0]
)

# named tuple for quantile stats
QuantileStat = namedtuple(
    'QuantileStat',
    field_names=['minimum','perc_1th',  'perc_5th', 'q1', 'median', 'q3', 'perc_95th', 'perc_99th', 'maximum', 'iqr', 'range']
)
# named typle for descriptive stats
DescStat = namedtuple(
    'DescStat',
    field_names=['mean', 'var', 'std', 'skew', 'kurt', 'mad', 'cv', 'normality_p']
)
# named tuple for extreme value analysis
XtremeStat = namedtuple(
    'XtremeStat',
    field_names=['lbound', 'rbound', 'lxtreme_mean', 'lxtreme_median', 'rxtreme_mean', 'rxtreme_median']
)


def serialize(v):
    if pd.isnull(v):
        return None
    if v == np.inf:
        return float('inf')
    if v == -np.inf:
        return -float('inf')
    if hasattr(v, 'item'):
        return v.item()
    else:
        return v 

class _BaseStat:
    def __init__(self):
        self.classname = self.__class__.__name__
    
    def fit(self, vector: pd.Series):
        raise NotImplemented("")
    
    def collect_attrs(self) -> dict:
        # gather serializable parameters for __init__
        raise NotImplemented("")

    def collect_params(self) -> dict:
        # gather serializable estimated parameters
        raise NotImplemented("")

    @classmethod
    def set_params(cls, obj, param_dict: dict):
        # set params estimated to the new cls object
        raise NotImplemented("")

    def to_dict(self) -> dict:
        if self.fitted_:
            return dict(
                colname = self.colname_,
                attr = self.collect_attrs(),
                params = self.collect_params()
            )
        else:
            raise Exception("Not fitted error")

    @classmethod
    def from_dict(cls, d: dict):
        c = cls(
            **d['attr']
        )
        c.colname_ = d['colname']
        cls.set_params(c, d['params'])
        c.fitted_ = True
        return c

    def generate(self, size: int = 100, seed: int = None, ignore_na:bool = False) -> pd.Series:
        raise NotImplemented("")

class CategStat(_BaseStat):
    def __init__(self, int_dtype: bool = False, max_categories: int = 25):
        super().__init__()
        self.fitted_ = False
        self.int_dtype_ = int_dtype
        self.max_categories_ = max_categories

    def fit(self, vector: pd.Series):
        self.colname_ = vector.name
        # total number of records
        self.total_ = len(vector)
        # missing rate
        missing = vector.isna()
        self.missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # num of unique value (dropped NA value)
        unique = len(vector.dropna().unique())
        self.unique_ = StatVar(
            value=serialize(unique), 
            perc=serialize(unique / self.total_)
        )
        # category counts without null
        self.vcounts_ = vector.value_counts(
            dropna = True,
            normalize = False
        ).fillna(0)
        self.vpercs_ = (self.vcounts_ / self.vcounts_.sum()).fillna(0)

        self.fitted_ = True

    def collect_attrs(self) -> dict:
        # gather serializable parameters for __init__
        return dict(
            int_dtype = self.int_dtype_,
            max_categories = self.max_categories_
        )

    def collect_params(self) -> dict:
        # gather serializable estimated parameters
        return dict(
            stats = dict(
                total = self.total_,
                missing = self.missing_._asdict(),
                unique = self.unique_._asdict()
            ),
            distribution = dict(
                vcounts = self.vcounts_.to_dict(),
                vpercs = self.vpercs_.to_dict(),
            )
        )

    @classmethod
    def set_params(cls, obj, param_dict: dict):
        # set params estimated to the new cls object
        # stats
        obj.total_ = param_dict['stats']['total']
        obj.missing_ = StatVar(**param_dict['stats']['missing'])
        obj.unique_ = StatVar(**param_dict['stats']['unique'])
        # distribution
        obj.vcounts_ = pd.Series(
            param_dict['distribution']['vcounts'], 
            name = obj.colname_
        )
        obj.vpercs_ = pd.Series(
            param_dict['distribution']['vpercs'], 
            name = obj.colname_
        )


    def generate(self, size: int = 100, seed: int = None, ignore_na:bool = False) -> pd.Series:
        if self.fitted_:
            rng = np.random.default_rng(seed = seed) # random generator
            choices = self.vpercs_.index.tolist()
            probs = self.vpercs_.values

            if not ignore_na:
                choices.append(pd.NA)
                probs = list(probs * (1 - self.missing_.perc))
                probs.append(self.missing_.perc)

            s = pd.Series(
                rng.choice(
                    a = choices,
                    size = size,
                    p = probs
                ),
                name = self.colname_,
            )
            if self.int_dtype_:
                s = s.astype('float').astype('Int64')
            return s
            
        else:
            raise Exception("Not fitted error")


class BinaryStat(CategStat):
    def __init__(self, pos_values: list, na_to_pos: bool = False, int_dtype: bool = False):
        super().__init__(int_dtype = int_dtype)
        self.pos_values_ = pos_values
        self.na_to_pos_ = na_to_pos

    def fit(self, vector: pd.Series):
        self.colname_ = vector.name
        # total number of records
        self.total_ = len(vector)
        # missing rate
        missing = vector.isna()
        self.missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # num of unique value (dropped NA value)
        unique = len(vector.dropna().unique())
        assert unique <= 2, f"Labeled as binary value, but {unique} categories are found"
        self.unique_ = StatVar(
            value=serialize(unique), 
            perc=serialize(unique / self.total_)
        )
        # category counts without null
        self.vcounts_ = vector.value_counts(
            dropna = True,
            normalize = False
        ).fillna(0)
        self.vpercs_ = (self.vcounts_ / self.vcounts_.sum()).fillna(0)

        # binary stat
        vector_binary = vector.copy()
        vector_binary[~vector_binary.isna()] = vector_binary[~vector_binary.isna()].apply(
            lambda v: 1 if v in self.pos_values_ else 0
        )
        vector_binary = vector_binary.fillna(int(self.na_to_pos_))
        self.binary_vcounts_ = vector_binary.value_counts(
            dropna = True,
            normalize = False
        ).fillna(0)
        self.binary_vpercs_ = (self.binary_vcounts_ / self.binary_vcounts_.sum()).fillna(0)

        self.fitted_ = True

    def collect_attrs(self) -> dict:
        # gather serializable parameters for __init__
        return dict(
            pos_values = self.pos_values_,
            int_dtype = self.int_dtype_,
            na_to_pos = self.na_to_pos_
        )

    def collect_params(self) -> dict:
        # gather serializable estimated parameters
        return dict(
            stats = dict(
                total = self.total_,
                missing = self.missing_._asdict(),
                unique = self.unique_._asdict()
            ),
            distribution = dict(
                vcounts = self.vcounts_.to_dict(),
                vpercs = self.vpercs_.to_dict(),
            ),
            binary_distribution = dict(
                binary_vcounts = self.binary_vcounts_.to_dict(),
                binary_vpercs = self.binary_vpercs_.to_dict(),
            )
        )

    @classmethod
    def set_params(cls, obj, param_dict: dict):
        # set params estimated to the new cls object
        # stats
        obj.total_ = param_dict['stats']['total']
        obj.missing_ = StatVar(**param_dict['stats']['missing'])
        obj.unique_ = StatVar(**param_dict['stats']['unique'])
        # distribution
        obj.vcounts_ = pd.Series(
            param_dict['distribution']['vcounts'], 
            name = obj.colname_
        )
        obj.vpercs_ = pd.Series(
            param_dict['distribution']['vpercs'], 
            name = obj.colname_
        )
        obj.binary_vcounts_ = pd.Series(
            param_dict['binary_distribution']['binary_vcounts'], 
            name = obj.colname_
        )
        obj.binary_vpercs_ = pd.Series(
            param_dict['binary_distribution']['binary_vpercs'], 
            name = obj.colname_
        )

class OrdinalCategStat(CategStat):
    def __init__(self, categories: List[Union[str, int]], int_dtype: bool = False):
        super().__init__(int_dtype = int_dtype)
        self.categories_ = categories

    def collect_attrs(self) -> dict:
        # gather serializable parameters for __init__
        return dict(
            int_dtype = self.int_dtype_,
            categories = self.categories_
        )
    
class NominalCategStat(CategStat):
    def to_ordinal(self, categories: List[Union[str, int]] = None) -> OrdinalCategStat:
        oc = OrdinalCategStat(
            categories = self.vcounts_.index.tolist() if categories is None else categories,
            int_dtype = self.int_dtype_
        )
        oc.colname_ = self.colname_
        self.set_params(
            self,
            param_dict = self.collect_params()
        )
        oc.fitted_ = self.fitted_
        return oc

def getNumericalStat(vector) -> Union[QuantileStat, DescStat]:
    vector = vector.astype('float')
    stat_quantile = QuantileStat(
        minimum = serialize(vector.min()),
        perc_1th = serialize(vector.quantile(0.01)),
        perc_5th = serialize(vector.quantile(0.05)),
        q1 = serialize(vector.quantile(0.25)),
        median = serialize(vector.quantile(0.5)),
        q3 = serialize(vector.quantile(0.75)),
        perc_95th = serialize(vector.quantile(0.95)),
        perc_99th = serialize(vector.quantile(0.99)),
        maximum = serialize(vector.max()),
        iqr = serialize(vector.quantile(0.75) - vector.quantile(0.25)),
        range = serialize(vector.max() - vector.min())
    )
    # descriptive statistics
    stat_descriptive = DescStat(
        mean = serialize(vector.mean()),
        var = serialize(vector.var()),
        std = serialize(vector.std()),
        skew = serialize(vector.skew()),
        kurt = serialize(vector.kurtosis()),
        mad = serialize(vector.mad()),
        cv = serialize(variation(vector, nan_policy='omit')),
        normality_p = serialize(kstest(vector.dropna(), cdf = 'norm')[1])
    )
    return stat_quantile, stat_descriptive

def getXtremeStat(vector, xtreme_method: Literal['iqr', 'quantile'] = 'iqr') -> XtremeStat:
    vector = vector.astype('float')
    if xtreme_method == 'iqr':
        q1 = vector.quantile(0.25)
        q3 = vector.quantile(0.75)
        iqr = q3 - q1
        lbound = q1 - 1.5 * iqr
        rbound = q3 + 1.5 * iqr
    
    elif xtreme_method == 'quantile':
        lbound = vector.quantile(0.01)
        rbound = vector.quantile(0.99)

    lvector = vector[vector < lbound]
    rvector = vector[vector > rbound]

    return XtremeStat(
        lbound = serialize(lbound),
        rbound = serialize(rbound),
        lxtreme_mean = serialize(lvector.mean()),
        lxtreme_median = serialize(lvector.median()),
        rxtreme_mean = serialize(rvector.mean()),
        rxtreme_median = serialize(rvector.median()),
    )

def log10pc(x):
    return np.where(x > 0, log10(x + 1), -log10(1 - x))

def exp10pc(x):
    return np.where(x > 0, exp10(x) - 1, -exp10(-x) + 1)

class NumericStat(_BaseStat):
    def __init__(self, setaside_zero: bool = False, log_scale: bool = False, xtreme_method: Literal['iqr', 'quantile'] = None, bins: int = 100):
        super().__init__()
        self.setaside_zero_ = setaside_zero
        self.log_scale_ = log_scale
        self.bins_ = bins
        self.xtreme_method_ = xtreme_method
        self.fitted_ = False

    def fit(self, vector: pd.Series):
        self.colname_ = vector.name
        # total number of records
        self.total_ = len(vector)
        # missing rate
        missing = vector.isna()
        self.missing_ = StatVar(
            value=serialize(missing.sum()), 
            perc=serialize(missing.mean())
        )
        # zero rate
        zeros = vector[vector == 0].count()
        self.zeros_ = StatVar(
            value=serialize(zeros), 
            perc=serialize(zeros / self.total_)
        )
        # infinite value
        infs_pos = vector[vector == np.inf].count()
        self.infs_pos_ = StatVar(
            value=serialize(infs_pos), 
            perc=serialize(infs_pos / self.total_)
        )
        infs_neg = vector[vector == -np.inf].count()
        self.infs_neg_ = StatVar(
            value=serialize(infs_neg), 
            perc=serialize(infs_neg / self.total_)
        )

        vector = vector.dropna()
        vector = vector[(vector < np.inf) & (vector > -np.inf)]
        if self.setaside_zero_:
            vector = vector[vector != 0]
        
        # xtreme value
        if self.xtreme_method_:
            self.xtreme_stat_ = getXtremeStat(vector, xtreme_method=self.xtreme_method_)
            vector_clean = vector[(vector >= self.xtreme_stat_.lbound) & (vector <= self.xtreme_stat_.rbound)]
            
        else:
            vector_clean = vector.copy()

        num_xtreme = len(vector) - len(vector_clean)
        self.xtreme_ = StatVar(
            value=serialize(num_xtreme), 
            perc=serialize(num_xtreme / self.total_)
        )

        # percentile & descriptive statistics
        self.stat_quantile_, self.stat_descriptive_= getNumericalStat(vector_clean)
        # histogram
        self.hist_, self.bin_edges_ = np.histogram(vector_clean, bins = self.bins_)

        if self.log_scale_:
            vector_log = pd.Series(log10pc(vector), name = self.colname_, dtype = 'float')

            # xtreme value
            if self.xtreme_method_:
                self.xtreme_stat_log_ = getXtremeStat(vector_log, xtreme_method=self.xtreme_method_)
                vector_clean_log = vector_log[(vector_log >= self.xtreme_stat_log_.lbound) & (vector_log <= self.xtreme_stat_log_.rbound)]
            else:
                vector_clean_log = vector_log.copy()

            num_xtreme_log = len(vector_log) - len(vector_clean_log)
            self.xtreme_log_ = StatVar(
                value=serialize(num_xtreme_log), 
                perc=serialize(num_xtreme_log / self.total_)
            )
            # percentile & descriptive statistics
            self.stat_quantile_log_, self.stat_descriptive_log_= getNumericalStat(vector_clean_log)
            # histogram
            self.hist_log_, self.bin_edges_log_ = np.histogram(vector_clean_log, bins = self.bins_)

        self.fitted_ = True

    def collect_attrs(self) -> dict:
        # gather serializable parameters for __init__
        return dict(
            setaside_zero = self.setaside_zero_,
            log_scale = self.log_scale_,
            bins = self.bins_,
            xtreme_method = self.xtreme_method_
        )

    def collect_params(self) -> dict:
        # gather serializable estimated parameters
        return dict(
            stats = dict(
                total = self.total_,
                missing = self.missing_._asdict(),
                zeros = self.zeros_._asdict(),
                infs_pos = self.infs_pos_._asdict(),
                infs_neg = self.infs_neg_._asdict(),
            ),
            xtreme_num = dict(
                origin = self.xtreme_._asdict() if hasattr(self, 'xtreme_') else None,
                log = self.xtreme_log_._asdict() if hasattr(self, 'xtreme_log_') else None
            ), # number of extreme values
            xtreme_stat = dict(
                origin = self.xtreme_stat_._asdict() if hasattr(self, 'xtreme_stat_') else None,
                log = self.xtreme_stat_log_._asdict() if hasattr(self, 'xtreme_stat_log_') else None
            ), # extreme value statistics
            stat_quantile = dict(
                origin = self.stat_quantile_._asdict(),
                log = self.stat_quantile_log_._asdict() if hasattr(self, 'stat_quantile_log_') else None
            ),
            stat_descriptive = dict(
                origin = self.stat_descriptive_._asdict(),
                log = self.stat_descriptive_log_._asdict() if hasattr(self, 'stat_descriptive_log_') else None
            ),
            histogram = dict(
                origin = dict(
                    hist = self.hist_.tolist(),
                    bin_edges = self.bin_edges_.tolist()
                ),
                log = dict(
                    hist = self.hist_log_.tolist() if hasattr(self, 'hist_log_') else None,
                    bin_edges = self.bin_edges_log_.tolist() if hasattr(self, 'bin_edges_log_') else None
                )
            )
        )

    @classmethod
    def set_params(cls, obj, param_dict: dict):
        # set params estimated to the new cls object
        obj.total_ = param_dict['stats']['total']
        obj.missing_ = StatVar(**param_dict['stats']['missing'])
        obj.zeros_ = StatVar(**param_dict['stats']['zeros'])
        obj.infs_pos_ = StatVar(**param_dict['stats']['infs_pos'])
        obj.infs_neg_ = StatVar(**param_dict['stats']['infs_neg'])
        obj.xtreme_ = StatVar(**param_dict['xtreme_num']['origin'])
        # origin statistics
        if obj.xtreme_method_ is not None:
            obj.xtreme_stat_ = XtremeStat(**param_dict['xtreme_stat']['origin'])
        obj.stat_quantile_ = QuantileStat(**param_dict['stat_quantile']['origin'])
        obj.stat_descriptive_ = DescStat(**param_dict['stat_descriptive']['origin'])
        obj.hist_ = np.array(param_dict['histogram']['origin']['hist'])
        obj.bin_edges_ = np.array(param_dict['histogram']['origin']['bin_edges'])
        # log statistics
        if obj.log_scale_:
            obj.xtreme_log_ = StatVar(**param_dict['xtreme_num']['log'])
            if obj.xtreme_method_ is not None:
                obj.xtreme_stat_log_ = XtremeStat(**param_dict['xtreme_stat']['log'])
            obj.stat_quantile_log_ = QuantileStat(**param_dict['stat_quantile']['log'])
            obj.stat_descriptive_log_ = DescStat(**param_dict['stat_descriptive']['log'])
            obj.hist_log_ = np.array(param_dict['histogram']['log']['hist'])
            obj.bin_edges_log_ = np.array(param_dict['histogram']['log']['bin_edges'])

    def generate(self, size: int = 100, seed: int = None, ignore_na:bool = False) -> pd.Series:
        if self.fitted_:
            if ignore_na:
                na_num = 0 # number of na values
                valid_num = size
                na_vs = []
            else:
                na_num = int(size * self.missing_.perc) # number of na values
                valid_num = size - na_num
                na_vs = [pd.NA] * na_num

            # size of zeros
            if self.setaside_zero_:
                zero_num = int(valid_num * self.zeros_.perc) # number of zeros
            else:
                zero_num = 0
            zero_vs = np.zeros(shape = zero_num)
            # size of infs
            inf_num = int(valid_num * self.infs_pos_.perc) # number of +inf
            neg_inf_num = int(valid_num * self.infs_neg_.perc) # number of -inf
            inf_vs = [np.inf] * inf_num
            neg_inf_vs = [-np.inf] * neg_inf_num

            # remaining valid nums
            valid_num = valid_num - zero_num - inf_num - neg_inf_num

            # valid numerical values
            if self.log_scale_:
                hist_dist = rv_histogram(
                    histogram=(self.hist_log_, self.bin_edges_log_),
                    seed = seed
                )
                valid_vs = exp10pc(hist_dist.rvs(size = valid_num))
            else:
                hist_dist = rv_histogram(
                    histogram=(self.hist_, self.bin_edges_),
                    seed = seed
                )
                valid_vs = hist_dist.rvs(size = valid_num)

            values = np.concatenate((valid_vs, inf_vs, neg_inf_vs, zero_vs, na_vs))
            np.random.shuffle(values)

            return pd.Series(
                values,
                name = self.colname_,
            ).astype(dtype='Float64')
        else:
            raise Exception("Not fitted error")


    
######### Univaraite Feature-Target correlation

def _combine_x_y(x: pd.Series, y: pd.Series, dropna: bool = True, combine_x_categ:bool = False) -> pd.DataFrame:
    df = pd.DataFrame({'x' : x.values, 'y' : y.values})
    if dropna:
        df = df.dropna(axis = 0, how = 'any')
    # categorical variable for x only:
    if combine_x_categ:
        vc = df['x'].value_counts(normalize = True)
        others = vc.iloc[20:].index  # only keep top 20 categories
        df.loc[df['x'].isin(others), 'x'] = 'Others'
    
    return df

class _BaseUniVarClfTargetCorr:
    def get_meta(self, x: pd.Series, y: pd.Series):
        self.colname_ = x.name
        self.yname_ = y.name
        self.ylabels_ = y.unique().astype('str').tolist()
    
    def fit(self, x: pd.Series, y: pd.Series):
        raise NotImplementedError("")
    
    def to_dict(self) -> dict:
        raise NotImplementedError("")
    
    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplementedError("")

class CategUniVarClfTargetCorr(_BaseUniVarClfTargetCorr):
    def __init__(self, ):
        self.fitted_ = False
    
    def fit(self, x: pd.Series, y: pd.Series):
        self.get_meta(x, y)
        
        df = _combine_x_y(
            x.astype('str'), 
            y.astype('str'), 
            dropna = True, 
            combine_x_categ = True # combine x's categories to be less than 20
        )
        
        # categorical:  p(x | y)
        self.p_x_y_ = {}
        for y in df['y'].unique():
            cs = df.loc[df['y'] == y, ['x']].groupby('x').size()
            d = pd.DataFrame({
                    'count' : cs.values, 
                    'perc' : cs  / cs.sum()
                }, 
                index = cs.index
            )
            self.p_x_y_[y] = d.fillna(0) #.to_dict(orient = 'index')
            
        # categorical:  p(y | x)  prob (event rate when binary clf) by category
        self.p_y_x_ = {}
        for x in df['x'].unique():
            cs = df.loc[df['x'] == x, ['y']].groupby('y').size()
            d = pd.DataFrame({
                    'count' : cs.values, 
                    'perc' : cs  / cs.sum()
                }, 
                index = cs.index
            )
            self.p_y_x_[x] = d.fillna(0) #.to_dict(orient = 'index')
            
        self.fitted_ = True
            
    def to_dict(self) -> dict:
        if self.fitted_:
            return dict(
                colname = self.colname_,
                yname = self.yname_,
                ylabels = self.ylabels_,
                attr = dict(),
                p_x_y = {
                    k: v.to_dict(orient = 'index')
                    for k, v in self.p_x_y_.items() 
                },
                p_y_x = {
                    k: v.to_dict(orient = 'index')
                    for k, v in self.p_y_x_.items() 
                },
            )
        else:
            raise Exception("Not fitted error")
        
    @classmethod
    def from_dict(cls, d: dict):
        c = cls(
            **d['attr']
        )
        c.colname_ = d['colname']
        c.yname_ = d['yname']
        c.ylabels_ = d['ylabels']
        
        c.p_x_y_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in d['p_x_y'].items()
        }
        c.p_y_x_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in d['p_y_x'].items()
        }
        return c

class NumericUniVarClfTargetCorr(_BaseUniVarClfTargetCorr):
    def __init__(self, ):
        self.fitted_ = False
    
    def fit(self, x: pd.Series, y: pd.Series):
        self.get_meta(x, y)
        
        df = _combine_x_y(
            x.astype('float'), 
            y.astype('str'), 
            dropna = False,
        )
        
        # numerical:  p(x | y)  distribution by target (boxplot)
        t = {'origin' : {}, 'log' : {}}
        for y in df['y'].unique():
            v = df.loc[df['y'] == y, 'x']
            ns = NumericStat(
                setaside_zero=False, 
                log_scale=True, 
                xtreme_method='iqr', 
                bins=100
            )
            ns.fit(v)
            
            t['origin'][y] = {
                'lbound': ns.xtreme_stat_.lbound,
                'q1' : ns.stat_quantile_.q1,
                'mean' : ns.stat_descriptive_.mean,
                'median' : ns.stat_quantile_.median,
                'q3' : ns.stat_quantile_.q3,
                'rbound': ns.xtreme_stat_.rbound,
            }
            t['log'][y] = {
                'lbound': ns.xtreme_stat_log_.lbound,
                'q1' : ns.stat_quantile_log_.q1,
                'mean' : ns.stat_descriptive_log_.mean,
                'median' : ns.stat_quantile_log_.median,
                'q3' : ns.stat_quantile_log_.q3,
                'rbound': ns.xtreme_stat_log_.rbound,
            }
            
        self.p_x_y_ = {
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
        self.p_y_x_ = mob.binning_table.build(add_totals=False)
        
        self.fitted_ = True
        
    def to_dict(self) -> dict:
        if self.fitted_:
            return dict(
                colname = self.colname_,
                yname = self.yname_,
                ylables = self.ylabels_,
                attr = dict(),
                p_x_y = {
                    k: v.to_dict(orient = 'index')
                    for k, v in self.p_x_y_.items() 
                },
                p_y_x = (
                    self.p_y_x_
                    .replace({np.nan: None})
                    .to_dict(orient = 'records')
                ),
            )
        else:
            raise Exception("Not fitted error")
        
    @classmethod
    def from_dict(cls, d: dict):
        c = cls(
            **d['attr']
        )
        c.colname_ = d['colname']
        c.yname_ = d['yname']
        c.ylabels_ = d['ylabels']
        
        c.p_x_y_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in d['p_x_y'].items()
        }
        c.p_y_x_ = pd.DataFrame.from_records(d['p_y_x'])
        return c


#########  Tabular

class _BaseTabularSerializable:
    def to_dict(self) -> dict:
        rs = {}
        for col, config in self.configs.items():
            r = config.to_dict()
            r['constructor'] = config.__class__.__name__
            rs[col] = r
        return rs
    
    @classmethod
    def from_dict(cls, r: dict):
        configs = {}
        for col, c in r.items():
            constructor = globals().get(c['constructor'])
            config = constructor.from_dict(c)
            configs[col] = config
        
        return cls(configs = configs)

    def dump(self, js_path: str):
        js = self.to_dict()
        toJSON(js = js, file = js_path)

    @classmethod
    def load(cls, js_path):
        js = loadJSON(file = js_path)
        return cls.from_dict(js)


class TabularStat(_BaseTabularSerializable):
    def __init__(self, configs: Dict[str, _BaseStat], n_jobs:int = 1):
        self.configs = configs
        self.n_jobs = n_jobs
        
    def get_categ_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, CategStat) or schema.classname in ('BinaryStat','NominalCategStat','OrdinalCategStat')]
    
    def get_nominal_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, NominalCategStat) or schema.classname == 'NominalCategStat']
    
    def get_binary_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, BinaryStat) or schema.classname == 'BinaryStat']
    
    def get_ordinal_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, OrdinalCategStat) or schema.classname == 'OrdinalCategStat']

    def get_numeric_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, NumericStat) or schema.classname == 'NumericStat']

    def fit(self, df: pd.DataFrame):
        # support multi-processor running
        cols = df.columns.intersection(self.configs.keys(), sort=False)
        jobs = (self._fit_one(col, df[col]) for col in cols)
        stats = parallel_run(jobs, n_jobs = self.n_jobs)
        self.configs = {col: stat for col, stat in zip(cols, stats)}

    @delayer
    def _fit_one(self, col: str, vector: pd.Series) -> _BaseStat:
        stat = self.configs[col]
        stat.fit(vector)
        return stat
    
    def generate(self, size: int = 100, seed: int = None, ignore_na:bool = False) -> pd.DataFrame:
        return pd.concat(
            [config.generate(
                size = size,
                seed = seed,
                ignore_na = ignore_na
            ) for col, config in self.configs.items()],
            axis = 1
        )


class TabularUniVarClfTargetCorr(_BaseTabularSerializable):
    def __init__(self, configs: Dict[str, _BaseUniVarClfTargetCorr], n_jobs:int = 1):
        self.configs = configs
        self.n_jobs = n_jobs
        
    def get_categ_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, CategUniVarClfTargetCorr)]

    def get_numeric_cols(self) -> List[str]:
        return [col for col, schema in self.configs.items() if isinstance(schema, NumericUniVarClfTargetCorr)]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # support multi-processor running
        cols = X.columns.intersection(self.configs.keys(), sort=False)
        jobs = (self._fit_one(col, X[col], y) for col in cols)
        stats = parallel_run(jobs, n_jobs = self.n_jobs)
        self.configs = {col: stat for col, stat in zip(cols, stats)}

    @delayer
    def _fit_one(self, col: str, x: pd.Series, y: pd.Series) -> _BaseUniVarClfTargetCorr:
        stat = self.configs[col]
        stat.fit(x, y)
        return stat