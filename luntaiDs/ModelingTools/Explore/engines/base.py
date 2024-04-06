from typing import Literal, Tuple
import numpy as np
import pandas as pd
from luntaiDs.ModelingTools.Explore.profiling import DescStat, QuantileStat, StatVar, XtremeStat
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, BinaryStatSummary, CategStatAttr, \
    CategStatSummary, NominalCategStatAttr, NominalCategStatSummary, NumericStatAttr, \
        NumericStatSummary, OrdinalCategStatAttr, OrdinalCategStatSummary


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
    
class _BaseNumericHelper:
    
    def get_descriptive_stat(self) -> DescStat:
        raise NotImplementedError("")
    
    def get_quantile_stat(self) -> QuantileStat:
        raise NotImplementedError("")        
    
    def get_xtreme_stat(self, xtreme_method: Literal["iqr", "quantile"] = "iqr") -> XtremeStat:
        raise NotImplementedError("")
        
    def get_histogram(self, n_bins:int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("")
    
class _BaseEDAEngine:
        
    def _fit_common_categ(self, colname: str, attr: CategStatAttr) -> CategStatSummary:
        """common categorical variable fitting can be reused by subclass categorical fitting

        :param str colname: column name
        :param CategStatAttr attr: categorical variable attribute object
        :return CategStatSummary: categorical variable summary object
        """
        raise NotImplementedError("")
        
    def fit_binary(self, colname: str, attr: BinaryStatAttr) -> BinaryStatSummary:
        """binary categorical variable fitting

        :param str colname: column name
        :param BinaryStatAttr attr: binary variable attribute object
        :return BinaryStatSummary: binary variable summary object
        """
        raise NotImplementedError("")
        
    def fit_ordinal(self, colname: str, attr: OrdinalCategStatAttr) -> OrdinalCategStatSummary:
        """ordinal categorical variable fitting

        :param str colname: column name
        :param OrdinalCategStatAttr attr: ordinal variable attribute object
        :return OrdinalCategStatSummary: ordinal variable summary object
        """
        # reuse common stat
        common_stat_summary = self._fit_common_categ(colname, attr)
        return OrdinalCategStatSummary(
            colname_ = common_stat_summary.colname_,
            total_ = common_stat_summary.total_,
            missing_ = common_stat_summary.missing_,
            unique_ = common_stat_summary.unique_,
            vcounts_ = common_stat_summary.vcounts_,
            vpercs_ = common_stat_summary.vpercs_,
        )
        
    def fit_nominal(self, colname: str, attr: NominalCategStatAttr) -> NominalCategStatSummary:
        """nominal categorical variable fitting

        :param str colname: column name
        :param NominalCategStatAttr attr: nominal variable attribute object
        :return NominalCategStatSummary: nominal variable summary object
        """
        # reuse common stat
        common_stat_summary = self._fit_common_categ(colname, attr)
        return NominalCategStatSummary(
            colname_ = common_stat_summary.colname_,
            total_ = common_stat_summary.total_,
            missing_ = common_stat_summary.missing_,
            unique_ = common_stat_summary.unique_,
            vcounts_ = common_stat_summary.vcounts_,
            vpercs_ = common_stat_summary.vpercs_,
        )
        
    def fit_numeric(self, colname: str, attr: NumericStatAttr) -> NumericStatSummary:
        """numeric variable fitting

        :param str colname: column name
        :param NumericStatAttr attr: numeric variable attribute object
        :return NumericStatSummary: numeric variable summary object
        """
        raise NotImplementedError("")