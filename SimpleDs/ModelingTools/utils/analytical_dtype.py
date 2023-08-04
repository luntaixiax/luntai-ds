# different from data dtype from data engineering perspective, each varaible should also have analytical dtype
from collections import OrderedDict
from typing import Dict, List

from CommonTools.dtyper import _Dtype


class _BaseAnlDtype:
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        self.dtype = dtype
        self.dimensional = dimensional # if it is static attribute (should not change over time)

class IdPolicyAnlDtype(_BaseAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)

class BinaryAnlDtype(_BaseAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)

class NumericAnlDtype(_BaseAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype = dtype, dimensional = dimensional)

class CardinalNumAnlDtype(NumericAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype = dtype, dimensional = dimensional)

class OrdinalNumAnlDtype(NumericAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype = dtype, dimensional = dimensional)


class CategoricalAnlDtype(_BaseAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)

class OrderedCategoricalAnlDtype(CategoricalAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)

class UnOrderedCategoricalAnlDtype(CategoricalAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)

class DateAnlDtype(_BaseAnlDtype):
    def __init__(self, dtype: _Dtype = None, dimensional: bool = False):
        super().__init__(dtype=dtype, dimensional=dimensional)


class AnalyticalSchema(OrderedDict):
    def __init__(self, schema_mapper: Dict[str, _BaseAnlDtype]):
        """create Schema object from predefined schema

        :param schema_mapper: {feature_name, feature_type} type should use dtyper._Dtype
        """
        super().__init__(**schema_mapper)

    def getColsByAnlDtype(self, anlDtype: type) -> List[str]:
        return list(filter(lambda t: isinstance(self[t], anlDtype), self.keys()))

