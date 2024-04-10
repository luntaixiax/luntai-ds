from typing import List, Literal, Tuple, Dict
import numpy as np
import pandas as pd
from luntaiDs.ModelingTools.Explore.summary import DescStat, QuantileStat, TabularStat, XtremeStat, \
    BaseStatAttr, BaseStatObj, BinaryStatAttr, BinaryStatSummary, CategStatAttr, CategStatSummary, \
        NominalCategStatAttr, NominalCategStatSummary, NumericStatAttr, \
        NumericStatSummary, OrdinalCategStatAttr, OrdinalCategStatSummary, \
        BinaryStatObj, OrdinalCategStatObj, NominalCategStatObj, NumericStatObj,\
        CategUniVarClfTargetCorr, NumericUniVarClfTargetCorr
from luntaiDs.ModelingTools.utils.parallel import delayer, parallel_run


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
    def get_columns(self) -> List[str]:
        """get all column list from given dataset

        :return List[str]: list of columns in the dataset
        """
        raise NotImplementedError("")
        
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
    
    def fit_binary_obj(self, colname: str, attr: BinaryStatAttr) -> BinaryStatObj:
        """binary categorical variable fitting

        :param str colname: column name
        :param BinaryStatAttr attr: binary variable attribute object
        :return BinaryStatObj: binary variable result object (attr + summary)
        """
        return BinaryStatObj(
            colname = colname,
            attr = attr,
            summary = self.fit_binary(colname, attr)
        )
        
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
        
    def fit_ordinal_obj(self, colname: str, attr: OrdinalCategStatAttr) -> OrdinalCategStatObj:
        """ordinal categorical variable fitting

        :param str colname: column name
        :param OrdinalCategStatAttr attr: ordinal variable attribute object
        :return OrdinalCategStatObj: ordinal variable result object (attr + summary)
        """
        return OrdinalCategStatObj(
            colname = colname,
            attr = attr,
            summary = self.fit_ordinal(colname, attr)
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
        
    def fit_nominal_obj(self, colname: str, attr: NominalCategStatAttr) -> NominalCategStatObj:
        """nominal categorical variable fitting

        :param str colname: column name
        :param NominalCategStatAttr attr: nominal variable attribute object
        :return NominalCategStatObj: nominal variable result object (attr + summary)
        """
        return NominalCategStatObj(
            colname = colname,
            attr = attr,
            summary = self.fit_nominal(colname, attr)
        )
        
    def fit_numeric(self, colname: str, attr: NumericStatAttr) -> NumericStatSummary:
        """numeric variable fitting

        :param str colname: column name
        :param NumericStatAttr attr: numeric variable attribute object
        :return NumericStatSummary: numeric variable summary object
        """
        raise NotImplementedError("")
    
    def fit_numeric_obj(self, colname: str, attr: NumericStatAttr) -> NumericStatObj:
        """numeric variable fitting

        :param str colname: _description_
        :param NumericStatAttr attr: numeric variable attribute object
        :return NumericStatObj: numeric variable result object (attr + summary)
        """
        return NumericStatObj(
            colname = colname,
            attr = attr,
            summary = self.fit_numeric(colname, attr)
        )
        
    
    @delayer
    def _fit_one(self, col: str, attr: BaseStatAttr) -> BaseStatObj:
        if isinstance(attr, NumericStatAttr):
            return self.fit_numeric_obj(col, attr)
        if isinstance(attr, NominalCategStatAttr):
            return self.fit_nominal_obj(col, attr)
        if isinstance(attr, OrdinalCategStatAttr):
            return self.fit_ordinal_obj(col, attr)
        if isinstance(attr, BinaryStatAttr):
            return self.fit_binary_obj(col, attr)
        else:
            raise TypeError("Only [NumericStatAttr/NominalCategStatAttr/OrdinalCategStatAttr/BinaryStatAttr] are supported")
    
    def fit(self, attrs: Dict[str, BaseStatAttr], n_jobs: int = 1) -> TabularStat:
        """fit a tabular dataset

        :param Dict[str, BaseStatAttr] attrs: dictionary of col:attribute pairs
        :param int n_jobs: control parallelism, defaults to 1
        :return TabularStat: tabular stat object
        """
        df_cols = self.get_columns()
        used_cols = [col for col in attrs.keys() if col in df_cols]
        jobs = (
            self._fit_one(col, attrs.get(col)) 
            for col in used_cols
        )
        summaries = parallel_run(jobs, n_jobs = n_jobs)
        return TabularStat(zip(used_cols, summaries))
    

    def fit_univarclf_categ(self, x_col: str, y_col: str) -> CategUniVarClfTargetCorr:
        """fit univariate correlation for classififer target and categorical variable

        :param str x_col: column of a categorical variable
        :param str y_col: column of the classifier target varaible
        :return CategUniVarClfTargetCorr: result object
        """
        raise NotImplementedError("")
    
    def fit_univarclf_numeric(self, x_col: str, y_col: str) -> NumericUniVarClfTargetCorr:
        """fit univariate correlation for classififer target and numeric variable

        :param str x_col: column of a numeric variable
        :param str y_col: column of the classifier target varaible
        :return NumericUniVarClfTargetCorr: result object
        """
        raise NotImplementedError("")