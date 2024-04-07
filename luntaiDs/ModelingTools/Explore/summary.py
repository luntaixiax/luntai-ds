from typing import Optional, Union, Any, Dict, List, Literal, TypeVar, MutableMapping
from collections import OrderedDict
from dataclasses import asdict, dataclass
import pandas as pd
import numpy as np

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
            constructor = self.__class__.__name__
            colname = self.colname, 
            attr = self.attr.serialize(), 
            summary = self.summary.serialize()
        )
    
    @classmethod
    def deserialize(cls, obj: dict):
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
class BinaryStatObj(BaseStatObj):
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
class OrdinalCategStatObj(BaseStatObj):
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
class NominalCategStatObj(BaseStatObj):
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
            or obj.classname in ("BinaryStatObj", "NominalCategStatObj", "OrdinalCategStatObj")
        ]

    def get_nominal_cols(self) -> list[str]:
        """Get nominal columns

        :return list[str]: list of nominal columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, NominalCategStatObj)
            or obj.classname == "NominalCategStatObj"
        ]

    def get_binary_cols(self) -> list[str]:
        """Get binary columns

        :return list[str]: list of binary columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, BinaryStatObj)
            or obj.classname == "BinaryStatObj"
        ]

    def get_ordinal_cols(self) -> list[str]:
        """Get ordinal columns

        :return list[str]: list of ordinal columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, OrdinalCategStatObj)
            or obj.classname == "OrdinalCategStatObj"
        ]

    def get_numeric_cols(self) -> list[str]:
        """Get numeric columns

        :return list[str]: list of numeric columns
        """
        return [
            col
            for col, obj in self.items()
            if isinstance(obj, NumericStatObj)
            or obj.classname == "NumericStatObj"
        ]