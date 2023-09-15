import pandas as pd
import numpy as np
from typing import Union, Dict, List, Literal
from collections import namedtuple
from scipy.special import exp10
from numpy import expm1, log1p, log10
from scipy.stats import rv_histogram
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
    field_names=['mean', 'var', 'std', 'skew', 'kurt', 'mad', 'cv']
)
# named tuple for extreme value analysis
XtremeStat = namedtuple(
    'XtremeStat',
    field_names=['lbound', 'rbound', 'lxtreme_mean', 'lxtreme_median', 'rxtreme_mean', 'rxtreme_median']
)

class _BaseStat:
    def fit(self, vector: pd.Series = None):
        raise NotImplemented("")

    def to_dict(self) -> dict:
        raise NotImplemented("")
    
    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplemented("")

    def generate(self, size: int = 100, seed: int = None, ignore_na:bool = False) -> pd.Series:
        raise NotImplemented("")

class CategStat(_BaseStat):
    def __init__(self, int_dtype: bool = False, binary: bool = False):
        self.fitted_ = False
        self.int_dtype_ = int_dtype
        self.binary_ = binary

    def fit(self, vector: pd.Series = None):
        self.colname_ = vector.name
        # total number of records
        self.total_ = len(vector)
        # missing rate
        missing = vector.isna()
        self.missing_ = StatVar(value=missing.sum().item(), perc=missing.mean())
        # num of unique value (dropped NA value)
        unique = len(vector.dropna().unique())
        if self.binary_:
            assert unique <= 2, f"Labeled as binary value, but {unique} categories are found"
        self.unique_ = StatVar(value=unique, perc=unique / self.total_)
        # category counts without null
        self.vcounts_ = vector.value_counts(
            dropna = True,
            normalize = False
        )
        self.vpercs_ = self.vcounts_ / self.vcounts_.sum()

        self.fitted_ = True

    def to_dict(self) -> dict:
        if self.fitted_:
            return dict(
                colname = self.colname_,
                attr = dict(
                    int_dtype = self.int_dtype_,
                    binary = self.binary_
                ),
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
        else:
            raise Exception("Not fitted error")
        
    @classmethod
    def from_dict(cls, d: dict):
        c = cls(
            int_dtype = d['attr']['int_dtype'],
            binary = d['attr']['binary']
        )
        c.colname_ = d['colname']
        # stats
        c.total_ = d['stats']['total']
        c.missing_ = StatVar(**d['stats']['missing'])
        c.unique_ = StatVar(**d['stats']['unique'])
        # distribution
        c.vcounts_ = pd.Series(
            d['distribution']['vcounts'], 
            name = d['colname']
        )
        c.vpercs_ = pd.Series(
            d['distribution']['vpercs'], 
            name = d['colname']
        )
        c.fitted_ = True
        return c

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
                s = s.astype('Int64')
            return s
            
        else:
            raise Exception("Not fitted error")

def getNumericalStat(vector) -> Union[QuantileStat, DescStat]:
    stat_quantile = QuantileStat(
        minimum = vector.min(),
        perc_1th = vector.quantile(0.01).item(),
        perc_5th = vector.quantile(0.05).item(),
        q1 = vector.quantile(0.25).item(),
        median = vector.quantile(0.5).item(),
        q3 = vector.quantile(0.75).item(),
        perc_95th = vector.quantile(0.95).item(),
        perc_99th = vector.quantile(0.99).item(),
        maximum = vector.max(),
        iqr = vector.quantile(0.75).item() - vector.quantile(0.25).item(),
        range = vector.max() - vector.min()
    )
    # descriptive statistics
    stat_descriptive = DescStat(
        mean = vector.mean(),
        var = vector.var(),
        std = vector.std(),
        skew = vector.skew(),
        kurt = vector.kurtosis(),
        mad = vector.mad(),
        cv = vector.std() / vector.mean()
    )
    return stat_quantile, stat_descriptive

def getXtremeStat(vector, xtreme_method: Literal['iqr', 'quantile'] = 'iqr') -> XtremeStat:
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
        lbound = lbound,
        rbound = rbound,
        lxtreme_mean = lvector.mean(),
        lxtreme_median = lvector.median(),
        rxtreme_mean = rvector.mean(),
        rxtreme_median = rvector.median(),
    )

def log10pc(x):
    return np.where(x > 0, log10(x + 1), -log10(1 - x))

def exp10pc(x):
    return np.where(x > 0, exp10(x) - 1, -exp10(-x) + 1)

class NumericStat(_BaseStat):
    def __init__(self, setaside_zero: bool = False, log_scale: bool = False, xtreme_method: Literal['iqr', 'quantile'] = None, bins: int = 100):
        self.setaside_zero_ = setaside_zero
        self.log_scale_ = log_scale
        self.bins_ = bins
        self.xtreme_method_ = xtreme_method
        self.fitted_ = False

    def fit(self, vector: pd.Series = None):
        self.colname_ = vector.name
        # total number of records
        self.total_ = len(vector)
        # missing rate
        missing = vector.isna()
        self.missing_ = StatVar(value=missing.sum().item(), perc=missing.mean())
        # zero rate
        zeros = vector[vector == 0].count().item()
        self.zeros_ = StatVar(value=zeros, perc=zeros / self.total_)
        # infinite value
        infs_pos = vector[vector == np.inf].count().item()
        self.infs_pos_ = StatVar(value=infs_pos, perc=infs_pos / self.total_)
        infs_neg = vector[vector == -np.inf].count().item()
        self.infs_neg_ = StatVar(value=infs_neg, perc=infs_neg / self.total_)

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
        self.xtreme_ = StatVar(value=num_xtreme, perc=num_xtreme / self.total_)

        # percentile & descriptive statistics
        self.stat_quantile_, self.stat_descriptive_= getNumericalStat(vector_clean)
        # histogram
        self.hist_, self.bin_edges_ = np.histogram(vector_clean, bins = self.bins_)

        if self.log_scale_:
            vector_log = pd.Series(log10pc(vector), name = self.colname_)

            # xtreme value
            if self.xtreme_method_:
                self.xtreme_stat_log_ = getXtremeStat(vector_log, xtreme_method=self.xtreme_method_)
                vector_clean_log = vector_log[(vector_log >= self.xtreme_stat_log_.lbound) & (vector_log <= self.xtreme_stat_log_.rbound)]
            else:
                vector_clean_log = vector_log.copy()

            num_xtreme_log = len(vector_log) - len(vector_clean_log)
            self.xtreme_log_ = StatVar(value=num_xtreme_log, perc=num_xtreme_log / self.total_)
            # percentile & descriptive statistics
            self.stat_quantile_log_, self.stat_descriptive_log_= getNumericalStat(vector_clean_log)
            # histogram
            self.hist_log_, self.bin_edges_log_ = np.histogram(vector_clean_log, bins = self.bins_)

        self.fitted_ = True

    def to_dict(self) -> dict:
        if self.fitted_:
            return dict(
                colname = self.colname_,
                attr = dict(
                    setaside_zero = self.setaside_zero_,
                    log_scale = self.log_scale_,
                    bins = self.bins_,
                    xtreme_method = self.xtreme_method_
                ),
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
        else:
            raise Exception("Not fitted error")

    @classmethod
    def from_dict(cls, d: dict):
        c = cls(
            setaside_zero = d['attr']['setaside_zero'],
            log_scale = d['attr']['log_scale'],
            bins = d['attr']['bins'],
            xtreme_method = d['attr']['xtreme_method']
        )
        c.colname_ = d['colname']
        # stats
        c.total_ = d['stats']['total']
        c.missing_ = StatVar(**d['stats']['missing'])
        c.zeros_ = StatVar(**d['stats']['zeros'])
        c.infs_pos_ = StatVar(**d['stats']['infs_pos'])
        c.infs_neg_ = StatVar(**d['stats']['infs_neg'])
        c.xtreme_ = StatVar(**d['xtreme_num']['origin'])
        # origin statistics
        if c.xtreme_method_ is not None:
            c.xtreme_stat_ = XtremeStat(**d['xtreme_stat']['origin'])
        c.stat_quantile_ = QuantileStat(**d['stat_quantile']['origin'])
        c.stat_descriptive_ = DescStat(**d['stat_descriptive']['origin'])
        c.hist_ = np.array(d['histogram']['origin']['hist'])
        c.bin_edges_ = np.array(d['histogram']['origin']['bin_edges'])
        # log statistics
        if c.log_scale_:
            c.xtreme_log_ = StatVar(**d['xtreme_num']['log'])
            if c.xtreme_method_ is not None:
                c.xtreme_stat_log_ = XtremeStat(**d['xtreme_stat']['log'])
            c.stat_quantile_log_ = QuantileStat(**d['stat_quantile']['log'])
            c.stat_descriptive_log_ = DescStat(**d['stat_descriptive']['log'])
            c.hist_log_ = np.array(d['histogram']['log']['hist'])
            c.bin_edges_log_ = np.array(d['histogram']['log']['bin_edges'])
        
        c.fitted_ = True
        return c

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


class TabularStat:
    def __init__(self, configs: Dict[str, _BaseStat], n_jobs:int = 1):
        self.configs = configs
        self.n_jobs = n_jobs

    def fit(self, df: pd.DataFrame):
        # support multi-processor running
        cols = self.configs.keys()
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
    