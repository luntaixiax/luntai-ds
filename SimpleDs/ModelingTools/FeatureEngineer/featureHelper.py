import logging
import pandas as pd
import numpy as np
from functools import partial
from typing import Union

class FeatureMeta:
    def __init__(self, features_meta: pd.DataFrame):
        """

        :param features_meta: pd.DataFrame([Varname, Vartype, ImputeStrategy, ImputeConst])

        Vartype: {Numerical, Binary, OrderedCateg, UnorderedCateg, IdPolicy}
        ImputeStrategy: {median, mean, most_frequent}
        ImputeConst: constant value to be imputed, e.g. 3, np.nan
        """
        self.features_meta = features_meta

    def subset(self, cols):
        self.features_meta = self.features_meta[self.features_meta['Varname'].isin(cols)]

    def getIdPolicyFeatures(self):
        # features that should not be included (typically Ids or policy reason (gender, geographic, marriage, racial, ...))
        return self.features_meta.loc[self.features_meta['Vartype'].isin(['IdPolicy']), 'Varname']

    def getNumericFeatures(self):
        return self.features_meta.loc[self.features_meta['Vartype'].isin(['Numerical']), 'Varname']

    def getCategFeatures(self):
        return self.features_meta.loc[
            self.features_meta['Vartype'].isin(['Binary', 'OrderedCateg', 'UnorderedCateg']), 'Varname']

    def getDiscreteFeatures(self):
        return self.features_meta.loc[self.features_meta['Vartype'].isin(['OrderedCateg', 'UnorderedCateg']), 'Varname']

    def getFeaturesByTypes(self, types: list):
        return self.features_meta.loc[self.features_meta['Vartype'].isin(types), 'Varname']

    def getImputeStrategyAndConst(self, col):
        return self.features_meta.loc[self.features_meta['Varname'] == col, 'ImputeStrategy'].iloc[0], \
               self.features_meta.loc[self.features_meta['Varname'] == col, 'ImputeConst'].iloc[0]

    def iterfeature(self):
        for idx, row in self.features_meta.iterrows():
            yield row['Varname'], row['Vartype'], row['ImputeStrategy'], row['ImputeConst']