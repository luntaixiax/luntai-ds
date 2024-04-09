from typing import List, Dict, Union, Literal, Optional, Any
from dataclasses import asdict, dataclass
import pandas as pd
import numpy as np
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MaxAbsScaler, \
    PowerTransformer, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.count import CountEncoder
from category_encoders.woe import WOEEncoder # bug
from luntaiDs.ModelingTools.Explore.summary import BaseStatObj, BinaryStatObj, NominalCategStatObj, \
    NumericStatObj, OrdinalCategStatAttr, OrdinalCategStatObj, OrdinalCategStatSummary, TabularStat, \
    exp10pc, log10pc
from luntaiDs.ModelingTools.FeatureEngineer.transformers import BucketCategValue, BucketCategByFsel, NamedTransformer, \
    MyImputer, BinaryConverter, OutlierClipper
from luntaiDs.ModelingTools.utils.support import make_present_col_selector


def intStringfy(X: pd.DataFrame) -> pd.DataFrame:
    return (
        X
        .astype('float')
        .astype('Int32')
        .astype('str')
        .replace(pd.NA, np.nan)
        .replace('<NA>', np.nan)
    )

def Stringfy(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype('str')

def filterZero(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace(0, np.nan)

def floatFy(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype('float')

@dataclass
class BasePreproc:
    def serialize(self) -> dict:
        return asdict(self)
    
    @classmethod
    def deserialize(cls, attr: dict):
        return cls(**attr)
    
    def compile_sklearn_pipeline(self, param: BaseStatObj) -> Pipeline:
        raise NotImplementedError("")


@dataclass
class BinaryPreproc(BasePreproc):
    def compile_sklearn_pipeline(self, param: BinaryStatObj) -> Pipeline:
        transformers = []

        if param.attr.int_dtype_:
            transformers.append(
                ('int_stringfy', FunctionTransformer(func=intStringfy))
            ) # convert to string type
        
        binarize = BinaryConverter(
            pos_values = list(str(v) for v in param.attr.pos_values_),
            keep_na = True
        )
        imputer = MyImputer(
            strategy = 'constant', 
            fill_value = int(param.attr.na_to_pos_), 
            add_indicator = False
        )

        return Pipeline(
            transformers + [
            ('binarize', binarize),
            ('impute', imputer)
        ])


@dataclass
class OrdinalCategPreproc(BasePreproc):
    impute: bool = True
    standardize: bool = False

    def compile_sklearn_pipeline(self, param: OrdinalCategStatObj) -> Pipeline:
        transformers = []

        if param.attr.int_dtype_:
            transformers.append(
                ('int_stringfy', FunctionTransformer(func=intStringfy))
            ) # convert to string type

        oe =  NamedTransformer(
            OrdinalEncoder(
                cols = [param.colname],
                mapping = [{
                    'col' : param.colname,
                    'mapping' : dict(zip(
                        map(str, param.attr.categories_), 
                        range(len(param.attr.categories_))
                    ))
                }],
                handle_unknown = 'value',
                handle_missing = 'value' if self.impute else 'return_nan'
            )
        )
        transformers.append(('ordinal', oe))
        
        if self.standardize:
            scaler = MaxAbsScaler()
            transformers.append(
                ('scale', NamedTransformer(scaler))
            )
        return Pipeline(transformers)

@dataclass
class NominalCategStatPreproc(BasePreproc):
    impute_value: Optional[str] = 'Other'
    bucket_strategy: Literal['freq', 'correlation'] = None
    encode_strategy: Optional[Literal['ohe', 'ce', 'woe']] = 'woe'
    
    def compile_sklearn_pipeline(self, param: NominalCategStatObj) -> Pipeline:
        transformers = []

        if self.impute_value is not None:
            if self.impute_value == 'most_frequent':
                imputer = MyImputer(
                    strategy = 'most_frequent',
                    add_indicator = False
                )
            else:
                imputer = MyImputer(
                    strategy = 'constant', 
                    fill_value = self.impute_value,
                    add_indicator = False
                )
            transformers.append(('impute', imputer))

        if param.attr.int_dtype_:
            transformers.append(
                ('int_stringfy', FunctionTransformer(func=intStringfy))
            ) # convert to string type
        else:
            transformers.append(
                ('stringfy', FunctionTransformer(func=Stringfy))
            ) # convert to string type

        if param.summary.unique_.value > param.attr.max_categories_:
            if self.bucket_strategy == 'freq':
                # freq threshold
                freq_threshold = param.summary.vpercs_.cumsum().iloc[param.attr.max_categories_ - 1]
                bucketizer = BucketCategValue(
                    threshold = freq_threshold,
                    handle_unknown = 'other' # will use `Other` indeed
                )
            else:
                bucketizer = BucketCategByFsel(
                    fsel = SelectKBest(
                        score_func = partial(
                            mutual_info_classif, 
                            discrete_features = True
                        ),
                        k = param.attr.max_categories_
                    ),
                    other_lvl_name = 'Other'
                )
            transformers.append(('bucketize', bucketizer))

        if self.encode_strategy is not None:
            if self.encode_strategy == 'ohe':
                encoder = OneHotEncoder(
                    sparse=False, 
                    handle_unknown='ignore'
                )
            elif self.encode_strategy == 'ce':
                encoder = CountEncoder(
                    handle_unknown='value', 
                    handle_missing='value',
                    normalize = True,
                    combine_min_nan_groups=True,
                )
            elif self.encode_strategy == 'woe':
                encoder = WOEEncoder(
                    handle_unknown='value', 
                    handle_missing='value',
                    randomized=True, # prevent overfitting
                    sigma=0.05
                )

            transformers.append(
                ('encode', NamedTransformer(encoder))
            )

        return Pipeline(transformers)


@dataclass
class NumericPreproc(BasePreproc):
    impute: bool = True
    normalize:bool = False
    standardize_strategy: Optional[Literal['robust', 'standard', 'maxabs']] = 'robust'
    
    def compile_sklearn_pipeline(self, param: NumericStatObj) -> Pipeline:
        transformers = []

        transformers.append(
            ('float', FunctionTransformer(func=floatFy))
        ) # convert to float type

        if param.attr.xtreme_method_ is not None:
            if param.attr.setaside_zero_:
                clipper = OutlierClipper(
                    strategy = param.attr.xtreme_method_,
                    quantile_range = (1, 99),
                    filter_func = filterZero
                )
            else:
                clipper = OutlierClipper(
                    strategy = param.attr.xtreme_method_,
                    quantile_range = (1, 99),
                )
            transformers.append(('clip', clipper))

        if self.impute:
            if param.attr.setaside_zero_:
                imputer = MyImputer(
                    strategy = 'constant',
                    fill_value = 0,
                    add_indicator = False
                )
            else:
                if param.summary.stat_descriptive_.skew is None:
                    strategy = 'median'
                elif abs(param.summary.stat_descriptive_.skew) > 0.5:
                    strategy = 'median'
                else:
                    strategy = 'mean'
                imputer = MyImputer(
                    strategy = strategy,
                    add_indicator = False
                )
            transformers.append(('impute', imputer))

        if param.attr.log_scale_:
            logger = NamedTransformer(
                FunctionTransformer(
                    func = log10pc, 
                    inverse_func = exp10pc
                )
            )
            transformers.append(('log', logger))

        if self.normalize:
            normalizer = NamedTransformer(
                PowerTransformer(
                    method = 'yeo-johnson',
                    standardize = False
                )
            )
            transformers.append(('normalize', normalizer))

        if self.standardize_strategy is not None:
            if self.standardize_strategy == 'robust':
                scaler = RobustScaler()
            elif self.standardize_strategy == 'maxabs':
                # will be between -1 and 1
                scaler = MaxAbsScaler()
            else:
                scaler = StandardScaler()
            transformers.append(('scale', NamedTransformer(scaler)))

        
        return Pipeline(transformers)


def get_preliminary_preprocess(ts: TabularStat) -> BaseEstimator:
    transformers = []    
    for col, stat in ts.items():
        if col in ts.get_nominal_cols():
            transformer = NominalCategStatPreproc(
                impute_value = 'Other',
                bucket_strategy = None,
                encode_strategy = 'woe'
            ).compile_sklearn_pipeline(stat)
        elif col in ts.get_ordinal_cols():
            transformer = OrdinalCategPreproc(
                impute = True,
                standardize = True
            ).compile_sklearn_pipeline(stat)
        elif col in ts.get_binary_cols():
            transformer = BinaryPreproc(
            ).compile_sklearn_pipeline(stat)
        elif col in ts.get_numeric_cols():
            transformer = NumericPreproc(
                impute = True,
                normalize = False,
                standardize_strategy = 'robust'
            ).compile_sklearn_pipeline(stat)
        
        transformers.append(
            (col, transformer, make_present_col_selector([col]))
        )
        
    pipe = NamedTransformer(
        ColumnTransformer(
            transformers,
            remainder='drop'
        )
    )
    return pipe


def get_mutual_info_preprocess(ts: TabularStat) -> BaseEstimator:
    transformers = []    
    for col, stat in ts.items():
        if col in ts.get_nominal_cols():
            # switch to Norminal stat
            stat_ord = OrdinalCategStatObj(
                colname = col,
                attr = OrdinalCategStatAttr(
                    categories_ = stat.summary.vcounts_.index.tolist(),
                    int_dtype_ = stat.attr.int_dtype_
                ),
                summary = OrdinalCategStatSummary(
                    colname_ = col,
                    total_ = stat.summary.total_,
                    missing_ = stat.summary.missing_,
                    unique_ = stat.summary.unique_,
                    vcounts_ = stat.summary.vcounts_,
                    vpercs_ = stat.summary.vpercs_,
                )
            )
            transformer = OrdinalCategPreproc(
                impute = True,
                standardize = False
            ).compile_sklearn_pipeline(stat_ord)
        elif col in ts.get_ordinal_cols():
            transformer = OrdinalCategPreproc(
                impute = True,
                standardize = False
            ).compile_sklearn_pipeline(stat)
        elif col in ts.get_binary_cols():
            transformer = BinaryPreproc(
            ).compile_sklearn_pipeline(stat)
        elif col in ts.get_numeric_cols():
            transformer = NumericPreproc(
                impute = True,
                normalize = False,
                standardize_strategy = 'robust'
            ).compile_sklearn_pipeline(stat)
        
        transformers.append(
            (col, transformer, make_present_col_selector([col]))
        )
        
    pipe = NamedTransformer(
        ColumnTransformer(
            transformers,
            remainder='drop'
        )
    )
    return pipe