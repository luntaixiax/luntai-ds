from typing import Union, Dict, List, Literal
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.count import CountEncoder
from luntaiDs.ModelingTools.CustomModel.linear import LinearClfStatsModelWrapper
    
def chi2_score_cross(X: pd.DataFrame) -> pd.DataFrame:
    categ_cols = X.columns
    result = []
    for c1 in categ_cols:
        for c2 in categ_cols:
            #if  c1 != c2:
            chi2, pvalue, dof, matrix = chi2_contingency(
                pd.crosstab(
                    X[c1],
                    X[c2]
                )
            )
            result.append((c1, c2, chi2, pvalue))
    chi_test_output = pd.DataFrame(
        result, 
        columns = ['var1', 'var2', 'chi2', 'pvalue']
    )
    return chi_test_output

def chi2_score_matrix(X: pd.DataFrame) -> pd.DataFrame:
    chi_test_output = chi2_score_cross(X)
    grid = chi_test_output.pivot(
        index='var1', 
        columns='var2', 
        values='pvalue'
    )
    grid.index.name = 'Feature'
    grid.columns.name = 'Features'
    return grid

def get_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    grid = df.corr()
    grid.index.name = 'Feature'
    grid.columns.name = 'Features'
    return grid