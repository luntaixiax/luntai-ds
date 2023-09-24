from typing import List, Dict, Union, Literal, Optional
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
from category_encoders.wrapper import PolynomialWrapper
from category_encoders.woe import WOEEncoder # bug
from ModelingTools.Explore.profiling import NominalCategStat, BinaryStat, OrdinalCategStat, NumericStat, TabularStat, \
    CategUniVarClfTargetCorr, NumericUniVarClfTargetCorr, exp10pc, log10pc
from ModelingTools.FeatureEngineer.transformers import BucketCategValue, BucketCategByFsel, NamedTransformer, \
    MyImputer, BinaryConverter, OutlierClipper
from ModelingTools.utils.support import make_present_col_selector


def intStringfy(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype('float').astype('Int32').astype('str').replace(pd.NA, np.nan).replace('<NA>', np.nan)

def Stringfy(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype('str')

def filterZero(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace(0, np.nan)

def floatFy(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype('float')

def nominal_categ_preprocess_pipe(
        cs: NominalCategStat, 
        impute_value: str = None,
        bucket_strategy: Literal['freq', 'correlation'] = 'freq',
        encode_strategy: Literal['ohe', 'ce', 'woe'] = None
    ) -> Pipeline:

    transformers = []

    if impute_value is not None:
        if impute_value == 'most_frequent':
            imputer = MyImputer(
                strategy = 'most_frequent',
                add_indicator = False
            )
        else:
            imputer = MyImputer(
                strategy = 'constant', 
                fill_value = impute_value,
                add_indicator = False
            )
        transformers.append(('impute', imputer))

    if cs.int_dtype_:
        transformers.append(('int_stringfy', FunctionTransformer(func=intStringfy))) # convert to string type
    else:
        transformers.append(('stringfy', FunctionTransformer(func=Stringfy))) # convert to string type

    if cs.unique_.value > cs.max_categories_:
        if bucket_strategy == 'freq':
            # freq threshold
            freq_threshold = cs.vpercs_.cumsum().iloc[cs.max_categories_ - 1]
            bucketizer = BucketCategValue(
                threshold = freq_threshold,
                handle_unknown = 'other' # will use `Other` indeed
            )
        else:
            bucketizer = BucketCategByFsel(
                fsel = SelectKBest(
                    score_func = partial(mutual_info_classif, discrete_features = True),
                    k = cs.max_categories_,
                ),
                other_lvl_name = 'Other'
            )
        transformers.append(('bucketize', bucketizer))

    if encode_strategy is not None:
        if encode_strategy == 'ohe':
            encoder = OneHotEncoder(
                sparse=False, 
                handle_unknown='ignore'
            )
        elif encode_strategy == 'ce':
            encoder = CountEncoder(
                handle_unknown='value', 
                handle_missing='value',
                normalize = True,
                combine_min_nan_groups=True,
            )
        elif encode_strategy == 'woe':
            encoder = WOEEncoder(
                handle_unknown='value', 
                handle_missing='value',
                randomized=True, # prevent overfitting
                sigma=0.05
            )

        transformers.append(('encode', NamedTransformer(encoder)))

    return Pipeline(transformers)


def binary_preprocess_pipe(
        bs: BinaryStat, 
    ) -> Pipeline:

    transformers = []

    if bs.int_dtype_:
        transformers.append(('int_stringfy', FunctionTransformer(func=intStringfy))) # convert to string type
    
    binarize = BinaryConverter(
        pos_values = list(str(v) for v in bs.pos_values_),
        keep_na = True
    )
    imputer = MyImputer(
        strategy = 'constant', 
        fill_value = int(bs.na_to_pos_), 
        add_indicator = False
    )

    return Pipeline(
        transformers + [
        ('binarize', binarize),
        ('impute', imputer)
    ])

def ordinal_categ_preprocess_pipe(
        os: OrdinalCategStat, 
        impute: bool = True,
        standardize: bool = False
    ) -> Pipeline:

    transformers = []

    if os.int_dtype_:
        transformers.append(('int_stringfy', FunctionTransformer(func=intStringfy))) # convert to string type

    oe =  NamedTransformer(
        OrdinalEncoder(
            cols = [os.colname_],
            mapping = [{
                'col' : os.colname_,
                'mapping' : dict(zip(
                    map(str, os.categories_), 
                    range(len(os.categories_))
                ))
            }],
            handle_unknown = 'value',
            handle_missing = 'value' if impute else 'return_nan'
        )
    )
    transformers.append(('ordinal', oe))
    
    if standardize:
        scaler = MaxAbsScaler()
        transformers.append(('scale', NamedTransformer(scaler)))
    return Pipeline(transformers)


def numeric_preprocess_pipe(
        ns: NumericStat,
        impute: bool = True,
        normalize:bool = False,
        standardize_strategy: Literal['robust', 'standard', 'maxabs'] = None,
    ) -> Pipeline:

    transformers = []

    transformers.append(('float', FunctionTransformer(func=floatFy))) # convert to float type

    if ns.xtreme_method_ is not None:
        if ns.setaside_zero_:
            clipper = OutlierClipper(
                strategy = ns.xtreme_method_,
                quantile_range = (1, 99),
                filter_func = filterZero
            )
        else:
            clipper = OutlierClipper(
                strategy = ns.xtreme_method_,
                quantile_range = (1, 99),
            )
        transformers.append(('clip', clipper))

    if impute:
        if ns.setaside_zero_:
            imputer = MyImputer(
                strategy = 'constant',
                fill_value = 0,
                add_indicator = False
            )
        else:
            if ns.stat_descriptive_.skew is None:
                strategy = 'median'
            elif abs(ns.stat_descriptive_.skew) > 0.5:
                strategy = 'median'
            else:
                strategy = 'mean'
            imputer = MyImputer(
                strategy = strategy,
                add_indicator = False
            )
        transformers.append(('impute', imputer))

    if ns.log_scale_:
        logger = NamedTransformer(
            FunctionTransformer(
                func = log10pc, 
                inverse_func = exp10pc
            )
        )
        transformers.append(('log', logger))

    if normalize:
        normalizer = NamedTransformer(
            PowerTransformer(
                method = 'yeo-johnson',
                standardize = False
            )
        )
        if ns.log_scale_:
            if ns.stat_descriptive_log_.normality_p < 0.01:
                transformers.append(('normalize', normalizer))
        else:
            if ns.stat_descriptive_.normality_p < 0.01:
                transformers.append(('normalize', normalizer))

    if standardize_strategy is not None:
        if standardize_strategy == 'robust':
            scaler = RobustScaler()
        elif standardize_strategy == 'maxabs':
            # will be between -1 and 1
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()
        transformers.append(('scale', NamedTransformer(scaler)))

    
    return Pipeline(transformers)



def get_preliminary_preprocess(ts: TabularStat) -> BaseEstimator:
    transformers = []    
    for col, stat in ts.configs.items():
        if col in ts.get_nominal_cols():
            transformer = nominal_categ_preprocess_pipe(
                cs = stat,
                impute_value = 'Other',
                bucket_strategy = None,
                encode_strategy = 'woe'
            )
        elif col in ts.get_ordinal_cols():
            transformer = ordinal_categ_preprocess_pipe(
                os = stat,
                impute = True,
                standardize = True
            )
        elif col in ts.get_binary_cols():
            transformer = binary_preprocess_pipe(
                bs = stat
            )
        elif col in ts.get_numeric_cols():
            transformer = numeric_preprocess_pipe(
                ns = stat,
                impute = True,
                normalize = False,
                standardize_strategy = 'robust'
            )
        
        transformers.append((col, transformer, make_present_col_selector([col])))
        
    pipe = NamedTransformer(
        ColumnTransformer(
            transformers,
            remainder='drop'
        )
    )
    return pipe


def get_mutual_info_preprocess(ts: TabularStat) -> BaseEstimator:
    transformers = []    
    for col, stat in ts.configs.items():
        if col in ts.get_nominal_cols():
            # switch to Norminal stat
            transformer = ordinal_categ_preprocess_pipe(
                os = stat.to_ordinal(),
                impute = True
            )
        elif col in ts.get_ordinal_cols():
            transformer = ordinal_categ_preprocess_pipe(
                os = stat,
                impute = True
            )
        elif col in ts.get_binary_cols():
            transformer = binary_preprocess_pipe(
                bs = stat
            )
        elif col in ts.get_numeric_cols():
            transformer = numeric_preprocess_pipe(
                ns = stat,
                impute = True,
                normalize = False,
                standardize_strategy = 'robust'
            )
        
        transformers.append((col, transformer, make_present_col_selector([col])))
        
    pipe = NamedTransformer(
        ColumnTransformer(
            transformers,
            remainder='drop'
        )
    )
    return pipe