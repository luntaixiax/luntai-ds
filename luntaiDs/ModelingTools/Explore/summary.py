from typing import Optional, Union, Any, Dict, List, Literal, MutableMapping
from collections import OrderedDict
from dataclasses import asdict, dataclass
import pandas as pd
import numpy as np
from scipy.special import exp10

@dataclass
class StatVar:
    """named tuple for value and percentage"""
    
    value: float = 0.0  # value
    perc: float = 0.0  # percentage

@dataclass
class QuantileStat:
    """named tuple for quantile stats"""

    minimum: float
    perc_1th: float
    perc_5th: float
    q1: float
    median: float
    q3: float
    perc_95th: float
    perc_99th: float
    maximum: float
    iqr: float
    range: float

@dataclass
class DescStat:
    """named tuple for descriptive stats"""

    mean: float
    var: float
    std: float
    skew: float
    kurt: float
    mad: float
    cv: float
    normality_p: Optional[float] = None
    
@dataclass
class XtremeStat:
    """named tuple for extreme value stats"""

    lbound: float
    rbound: float
    lxtreme_mean: float
    lxtreme_median: float
    rxtreme_mean: float
    rxtreme_median: float
    
'''Base stat statistics'''
@dataclass    
class BaseStatAttr:
    def serialize(self) -> dict:
        return asdict(self)
    
    @classmethod
    def deserialize(cls, attr: dict):
        return cls(**attr)

@dataclass    
class BaseStatSummary:
    def serialize(self) -> dict:
        raise NotImplementedError("")
    
    @classmethod
    def deserialize(cls, summary: dict):
        raise NotImplementedError("") 
    
@dataclass
class BaseStatObj:
    colname: str
    attr: Any
    summary: Any
    
    def serialize(self) -> dict:
        return dict(
            constructor = self.__class__.__name__,
            colname = self.colname, 
            attr = self.attr.serialize(), 
            summary = self.summary.serialize()
        )
    
    @classmethod
    def deserialize(cls, obj: dict):
        raise NotImplementedError("")
    
    def generate(self, size: int = 100, seed: Optional[int] = None, 
                 ignore_na: bool = False) -> pd.Series:
        """Generate fake data based on learnt distribution

        :param int size: number of samples, defaults to 100
        :param Optional[int] seed: random seed, defaults to None
        :param bool ignore_na: whether to ignore NA value (fill NA with values), defaults to False
        :return pd.Series: generated pandas series
        """
        raise NotImplementedError("")
    
'''CategStat'''

@dataclass
class CategStatAttr(BaseStatAttr):
    pass
    
@dataclass
class CategStatSummary(BaseStatSummary):
    colname_: str
    total_: int
    missing_: StatVar
    unique_: StatVar
    vcounts_: pd.Series
    vpercs_: pd.Series
    
    def serialize(self) -> dict:
        return dict(
            colname=self.colname_,
            stats=dict(
                total=self.total_, 
                missing=asdict(self.missing_), 
                unique=asdict(self.unique_)
            ),
            distribution=dict(
                vcounts=self.vcounts_.to_dict(),
                vpercs=self.vpercs_.to_dict(),
            )
        )
    
    @classmethod
    def deserialize(cls, summary: dict):
        return cls(
            colname_ = summary.get('colname', 'NONAME'),
            total_ = summary["stats"]["total"],
            missing_ = StatVar(**summary["stats"]["missing"]),
            unique_ = StatVar(**summary["stats"]["unique"]),
            vcounts_ = pd.Series(
                summary["distribution"]["vcounts"], 
                name=summary.get('colname', 'NONAME')
            ),
            vpercs_ = pd.Series(
                summary["distribution"]["vpercs"], 
                name=summary.get('colname', 'NONAME')
            ),
        )

@dataclass
class CategStatObj(BaseStatObj):
    colname: str
    attr: CategStatAttr
    summary: CategStatSummary
    
    def generate(self, size: int = 100, seed: Optional[int] = None, 
                 ignore_na: bool = False
    ) -> pd.Series:
        """Generate fake data based on learnt distribution

        :param int size: number of samples, defaults to 100
        :param Optional[int] seed: random seed, defaults to None
        :param bool ignore_na: whether to ignore NA value (fill NA with values), defaults to False
        :return pd.Series: generated pandas series
        """
        rng = np.random.default_rng(seed=seed)  # random generator
        choices = self.summary.vpercs_.index.tolist()
        probs = self.summary.vpercs_.values

        if not ignore_na:
            choices.append(pd.NA)
            probs = list(probs * (1 - self.summary.missing_.perc))
            probs.append(self.summary.missing_.perc)

        s = pd.Series(
            rng.choice(a=choices, size=size, p=probs),
            name=self.colname,
        )
        if self.attr.int_dtype_:
            s = s.astype("float").astype("Int64")
        return s

    
    
'''BinaryStat'''
@dataclass
class BinaryStatAttr(CategStatAttr):
    pos_values_: list
    na_to_pos_: bool = False
    int_dtype_: bool = False
    
@dataclass
class BinaryStatSummary(CategStatSummary):
    binary_vcounts_: pd.Series
    binary_vpercs_: pd.Series
    
    def serialize(self) -> dict:
        return dict(
            colname=self.colname_,
            stats=dict(
                total=self.total_, 
                missing=asdict(self.missing_), 
                unique=asdict(self.unique_)
            ),
            distribution=dict(
                vcounts=self.vcounts_.to_dict(),
                vpercs=self.vpercs_.to_dict(),
            ),
            binary_distribution=dict(
                binary_vcounts=self.binary_vcounts_.to_dict(),
                binary_vpercs=self.binary_vpercs_.to_dict(),
            ),
        )
    
    @classmethod
    def deserialize(cls, summary: dict):
        return cls(
            colname_ = summary.get('colname', 'NONAME'),
            total_ = summary["stats"]["total"],
            missing_ = StatVar(**summary["stats"]["missing"]),
            unique_ = StatVar(**summary["stats"]["unique"]),
            vcounts_ = pd.Series(
                summary["distribution"]["vcounts"], 
                name=summary.get('colname', 'NONAME')
            ),
            vpercs_ = pd.Series(
                summary["distribution"]["vpercs"], 
                name=summary.get('colname', 'NONAME')
            ),
            binary_vcounts_ = pd.Series(
                summary["binary_distribution"]["binary_vcounts"], 
                name=summary.get('colname', 'NONAME')
            ),
            binary_vpercs_ = pd.Series(
                summary["binary_distribution"]["binary_vpercs"], 
                name=summary.get('colname', 'NONAME')
            )
        )
          
    
@dataclass
class BinaryStatObj(CategStatObj):
    colname: str
    attr: BinaryStatAttr
    summary: BinaryStatSummary
    
    @classmethod
    def deserialize(cls, obj: dict):
        return cls(
            colname = obj.get('colname'),
            attr = BinaryStatAttr.deserialize(obj.get('attr')),
            summary = BinaryStatSummary.deserialize(obj.get('summary'))
        )

'''OrdinalCategStat'''
@dataclass
class OrdinalCategStatAttr(CategStatAttr):
    categories_: List[Union[str, int]]
    int_dtype_: bool = False
    
@dataclass
class OrdinalCategStatSummary(CategStatSummary):
    pass
    
@dataclass
class OrdinalCategStatObj(CategStatObj):
    colname: str
    attr: OrdinalCategStatAttr
    summary: OrdinalCategStatSummary
    
    @classmethod
    def deserialize(cls, obj: dict):
        return cls(
            colname = obj.get('colname'),
            attr = OrdinalCategStatAttr.deserialize(obj.get('attr')),
            summary = OrdinalCategStatSummary.deserialize(obj.get('summary'))
        )
    
    
'''NominalCategStat'''
@dataclass
class NominalCategStatAttr(CategStatAttr):
    int_dtype_: bool = False
    max_categories_: int = 25
    
@dataclass
class NominalCategStatSummary(CategStatSummary):
    pass

@dataclass
class NominalCategStatObj(CategStatObj):
    colname: str
    attr: NominalCategStatAttr
    summary: NominalCategStatSummary
    
    @classmethod
    def deserialize(cls, obj: dict):
        return cls(
            colname = obj.get('colname'),
            attr = NominalCategStatAttr.deserialize(obj.get('attr')),
            summary = NominalCategStatSummary.deserialize(obj.get('summary'))
        )

'''NumericStat'''
def safe_asdict(v) -> Optional[dict]:
    if v is None:
        return
    return asdict(v)

def safe_tolist(v: Optional[np.ndarray]) -> Optional[list]:
    if v is None:
        return
    return v.tolist()

def safe_toarray(v: Optional[list]) -> Optional[np.ndarray]:
    if v is None:
        return
    return np.array(v)

def safe_to_stv(params: Optional[dict]) -> Optional[StatVar]:
    if params is None:
        return
    return StatVar(**params)

def safe_to_qst(params: Optional[dict]) -> Optional[QuantileStat]:
    if params is None:
        return
    return QuantileStat(**params)

def safe_to_dst(params: Optional[dict]) -> Optional[DescStat]:
    if params is None:
        return
    return DescStat(**params)

def safe_to_xst(params: Optional[dict]) -> Optional[XtremeStat]:
    if params is None:
        return
    return XtremeStat(**params)

def log10pc(x: np.ndarray) -> np.ndarray:
    """Do log10p transform on both positive and negative range

    :param np.ndarray x: original array
    :return np.ndarray: transformed array
    """
    return np.where(x > 0, np.log10(x + 1), -np.log10(1 - x))

def exp10pc(x: np.ndarray) -> np.ndarray:
    """Do exp10m transform on both positive and negative range

    :param np.ndarray x: original array
    :return np.ndarray: transformed array
    """
    return np.where(x > 0, exp10(x) - 1, -exp10(-x) + 1)
    

@dataclass
class NumericStatAttr(BaseStatAttr):
    setaside_zero_: bool = False
    log_scale_: bool = False
    xtreme_method_: Optional[Literal["iqr", "quantile"]] = None
    bins_: int = 100
    
@dataclass
class NumericStatSummary(BaseStatSummary):
    colname_: str
    total_: int
    missing_: StatVar
    zeros_: StatVar
    infs_pos_: StatVar
    infs_neg_: StatVar
    stat_quantile_: QuantileStat
    stat_descriptive_: DescStat
    hist_: np.ndarray
    bin_edges_: np.ndarray
    xtreme_: StatVar
    xtreme_stat_: Optional[XtremeStat] = None
    xtreme_stat_log_: Optional[XtremeStat] = None
    xtreme_log_: Optional[StatVar] = None
    stat_quantile_log_: Optional[QuantileStat] = None
    stat_descriptive_log_: Optional[DescStat] = None
    hist_log_: Optional[np.ndarray] = None
    bin_edges_log_: Optional[np.ndarray] = None
    
    def serialize(self) -> dict:
        return dict(
            colname=self.colname_,
            stats=dict(
                total=self.total_,
                missing=asdict(self.missing_),
                zeros=asdict(self.zeros_),
                infs_pos=asdict(self.infs_pos_),
                infs_neg=asdict(self.infs_neg_),
            ),
            xtreme_num=dict(
                origin=safe_asdict(self.xtreme_),
                log=safe_asdict(self.xtreme_log_),
            ),  # number of extreme values
            xtreme_stat=dict(
                origin=safe_asdict(self.xtreme_stat_),
                log=safe_asdict(self.xtreme_stat_log_),
            ),  # extreme value statistics
            stat_quantile=dict(
                origin=asdict(self.stat_quantile_),
                log=safe_asdict(self.stat_quantile_log_),
            ),
            stat_descriptive=dict(
                origin=asdict(self.stat_descriptive_),
                log=safe_asdict(self.stat_descriptive_log_),
            ),
            histogram=dict(
                origin=dict(
                    hist=self.hist_.tolist(), 
                    bin_edges=self.bin_edges_.tolist()
                ),
                log=dict(
                    hist=safe_tolist(self.hist_log_),
                    bin_edges=safe_tolist(self.bin_edges_log_)
                ),
            ),
        )
    
    @classmethod
    def deserialize(cls, summary: dict):
        return cls(
            colname_ = summary.get('colname', 'NONAME'),
            total_ = summary["stats"]["total"],
            missing_ = StatVar(**summary["stats"]["missing"]),
            zeros_ = StatVar(**summary["stats"]["zeros"]),
            infs_pos_ = StatVar(**summary["stats"]["infs_pos"]),
            infs_neg_ = StatVar(**summary["stats"]["infs_neg"]),
            xtreme_ = StatVar(**summary["xtreme_num"]["origin"]),
            
            stat_quantile_ = QuantileStat(**summary["stat_quantile"]["origin"]),
            stat_descriptive_ = DescStat(**summary["stat_descriptive"]["origin"]),
            hist_ = np.array(summary["histogram"]["origin"]["hist"]),
            bin_edges_ = np.array(summary["histogram"]["origin"]["bin_edges"]),
            
            # need
            xtreme_stat_ = safe_to_xst(summary["xtreme_stat"]["origin"]),
            xtreme_log_ = safe_to_stv(summary["xtreme_num"]["log"]),
            xtreme_stat_log_ = safe_to_xst(summary["xtreme_stat"]["log"]),
            stat_quantile_log_ = safe_to_qst(summary["stat_quantile"]["log"]),
            stat_descriptive_log_ = safe_to_dst(summary["stat_descriptive"]["log"]),
            hist_log_ = safe_toarray(summary["histogram"]["log"]["hist"]),
            bin_edges_log_ = safe_toarray(summary["histogram"]["log"]["bin_edges"])
        )
   
@dataclass
class NumericStatObj(BaseStatObj):
    colname: str
    attr: NumericStatAttr
    summary: NumericStatSummary
    
    @classmethod
    def deserialize(cls, obj: dict):
        return cls(
            colname = obj.get('colname'),
            attr = NumericStatAttr.deserialize(obj.get('attr')),
            summary = NumericStatSummary.deserialize(obj.get('summary'))
        )
    
    def generate(self, size: int = 100, seed: Optional[int] = None, 
                 ignore_na: bool = False) -> pd.Series:
        """Generate fake data based on learnt distribution

        :param int size: number of samples, defaults to 100
        :param Optional[int] seed: random seed, defaults to None
        :param bool ignore_na: whether to ignore NA value (fill NA with values), defaults to False
        :return pd.Series: generated pandas series
        """
        from scipy.stats import rv_histogram
        
        if ignore_na:
            na_num = 0  # number of na values
            valid_num = size
            na_vs = []
        else:
            na_num = int(size * self.summary.missing_.perc)  # number of na values
            valid_num = size - na_num
            na_vs = [pd.NA] * na_num

        # size of zeros
        if self.attr.setaside_zero_:
            zero_num = int(valid_num * self.summary.zeros_.perc)  # number of zeros
        else:
            zero_num = 0
        zero_vs = np.zeros(shape=zero_num)
        # size of infs
        inf_num = int(valid_num * self.summary.infs_pos_.perc)  # number of +inf
        neg_inf_num = int(valid_num * self.summary.infs_neg_.perc)  # number of -inf
        inf_vs = [np.inf] * inf_num
        neg_inf_vs = [-np.inf] * neg_inf_num

        # remaining valid nums
        valid_num = valid_num - zero_num - inf_num - neg_inf_num

        # valid numerical values
        if self.attr.log_scale_:
            hist_dist = rv_histogram(
                histogram=(self.summary.hist_log_, self.summary.bin_edges_log_), 
                seed=seed
            )
            valid_vs = exp10pc(hist_dist.rvs(size=valid_num, random_state = seed))
        else:
            hist_dist = rv_histogram(
                histogram=(self.summary.hist_, self.summary.bin_edges_), 
                seed=seed
            )
            valid_vs = hist_dist.rvs(size=valid_num, random_state = seed)

        values = np.concatenate((valid_vs, inf_vs, neg_inf_vs, zero_vs, na_vs))
        np.random.seed(seed)
        np.random.shuffle(values)
        np.random.seed(None) # reset

        return pd.Series(
            values,
            name=self.colname,
        ).astype(dtype="Float64")
        

'''Tabular'''

class TabularStat(
        OrderedDict, 
        MutableMapping[str, BaseStatObj]
    ):
    CONSTRUCTORS = {
        NumericStatObj.__name__: NumericStatObj,
        NominalCategStatObj.__name__: NominalCategStatObj,
        OrdinalCategStatObj.__name__: OrdinalCategStatObj,
        BinaryStatObj.__name__: BinaryStatObj,
    }
    """Tabular stat for multiple columns"""
    
    def serialize(self) -> dict:
        return {
            col: obj.serialize() for col, obj in self.items()
        }
    
    @classmethod
    def deserialize(cls, stat_dict: dict):
        r = OrderedDict()
        for col, obj in stat_dict.items():
            c = cls.CONSTRUCTORS.get(obj['constructor']) # constructor
            r[col] = c.deserialize(obj)
        return r

    def get_categ_cols(self) -> list[str]:
        """Get categorical columns (include binary, nominal, ordinal)

        :return list[str]: list of categ columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, (BinaryStatObj, NominalCategStatObj, OrdinalCategStatObj))
            or obj.__class__.__name__ in ("BinaryStatObj", "NominalCategStatObj", "OrdinalCategStatObj")
        ]

    def get_nominal_cols(self) -> list[str]:
        """Get nominal columns

        :return list[str]: list of nominal columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, NominalCategStatObj)
            or obj.__class__.__name__ == "NominalCategStatObj"
        ]

    def get_binary_cols(self) -> list[str]:
        """Get binary columns

        :return list[str]: list of binary columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, BinaryStatObj)
            or obj.__class__.__name__ == "BinaryStatObj"
        ]

    def get_ordinal_cols(self) -> list[str]:
        """Get ordinal columns

        :return list[str]: list of ordinal columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, OrdinalCategStatObj)
            or obj.__class__.__name__ == "OrdinalCategStatObj"
        ]

    def get_numeric_cols(self) -> list[str]:
        """Get numeric columns

        :return list[str]: list of numeric columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, NumericStatObj)
            or obj.__class__.__name__ == "NumericStatObj"
        ]
        
    def generate(self, size: int = 100, seed: Optional[int] = None, 
                 ignore_na: bool = False
    ) -> pd.DataFrame:
        """Generate fake data based on learnt distribution

        :param int size: number of samples, defaults to 100
        :param Optional[int] seed: random seed, defaults to None
        :param bool ignore_na: whether to ignore NA value (fill NA with values), defaults to False
        :return pd.DataFrame: generated fake data
        """
        return pd.concat(
            [
                obj.generate(
                    size=size, 
                    seed=seed, 
                    ignore_na=ignore_na
                )
                for col, obj in self.items()
            ],
            axis=1,
        )
        

'''Univaraite Corr'''

@dataclass
class BaseUniVarClfTargetCorr:
    colname_: str
    yname_: str
    ylabels_: List[str]
    
@dataclass
class CategUniVarClfTargetCorr(BaseUniVarClfTargetCorr):
    p_x_y_: Dict[str, pd.DataFrame]
    p_y_x_: Dict[str, pd.DataFrame]
    
    def serialize(self) -> dict:
        return dict(
            constructor = self.__class__.__name__,
            colname = self.colname_,
            yname = self.yname_,
            ylables = self.ylabels_,
            p_x_y = {
                k: v.to_dict(orient = 'index')
                for k, v in self.p_x_y_.items() 
            },
            p_y_x = {
                k: v.to_dict(orient = 'index')
                for k, v in self.p_y_x_.items() 
            },
        )

    @classmethod
    def deserialize(cls, stat_dict: dict):
        c = cls()
        c.colname_ = stat_dict['colname']
        c.yname_ = stat_dict['yname']
        c.ylabels_ = stat_dict['ylabels']
        c.p_x_y_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in stat_dict['p_x_y'].items()
        }
        c.p_y_x_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in stat_dict['p_y_x'].items()
        }
        return c

    
@dataclass
class NumericUniVarClfTargetCorr(BaseUniVarClfTargetCorr):
    p_x_y_: Dict[str, pd.DataFrame]
    p_y_x_: pd.DataFrame
    
    def serialize(self) -> dict:
        return dict(
            constructor = self.__class__.__name__,
            colname = self.colname_,
            yname = self.yname_,
            ylables = self.ylabels_,
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
    
    @classmethod
    def deserialize(cls, stat_dict: dict):
        c = cls()
        c.colname_ = stat_dict['colname']
        c.yname_ = stat_dict['yname']
        c.ylabels_ = stat_dict['ylabels']
        
        c.p_x_y_ = {
            k: pd.DataFrame.from_dict(v, orient = 'index')
            for k, v in stat_dict['p_x_y'].items()
        }
        c.p_y_x_ = pd.DataFrame.from_records(stat_dict['p_y_x'])
        return c