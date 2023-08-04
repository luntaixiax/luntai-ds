import logging
from collections import OrderedDict
from enum import Enum
from typing import Dict
import yaml
import numpy as np


#### Internal representation types
class _Dtype:
    def collect_args(self) -> dict:
        return {}

    def to_config(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "args": self.collect_args()
        }

    @classmethod
    def from_config(cls, config: dict):
        template = config['type']
        params = config['args']
        mapping = {
            'bool' : Bool,
            'integer' : Integer,
            'float' : Float,
            'decimal' : Decimal,
            'string' : String,
            'datet' : DateT,
            'datetimet': DateTimeT,
            'array': Array,
            'map': Map
        }
        block = mapping.get(template.lower()) # the class of relavant dtype
        return block.from_args(params)

class _BaseDtype(_Dtype):
    def __init__(self, nullable:bool = True):
        self.nullable = bool(nullable)

    @classmethod
    def from_args(cls, args:dict):
        # de-serialize
        return cls(**args)

    def __repr__(self):
        return self.__str__()

    def collect_args(self) -> dict:
        return dict(nullable = self.nullable)

class Integer(_BaseDtype):
    def __init__(self, nullable:bool = True, precision: int = 32, signed:bool = True):
        super().__init__(nullable)
        self.precision = int(precision)
        self.signed = bool(signed)

    def __str__(self):
        # will use numpy type
        return f"Integer{self.precision}(null={self.nullable}, signed={self.signed})"

    def collect_args(self) -> dict:
        return dict(nullable = self.nullable, precision = self.precision, signed = self.signed)


class Decimal(_BaseDtype):
    def __init__(self, nullable: bool = True, precision: int = 10, scale: int = 2):
        super().__init__(nullable)
        self.precision = int(precision)
        self.scale = int(scale)

    def __str__(self):
        return f"Decimal({self.precision}, {self.scale}, null={self.nullable})"

    def collect_args(self) -> dict:
        return dict(nullable = self.nullable, precision = self.precision, scale = self.scale)

class Float(_BaseDtype):
    def __init__(self, nullable: bool = True, precision: int = 32):
        super().__init__(nullable)
        self.precision = int(precision)

    def __str__(self):
        return f"Float{self.precision}(null={self.nullable})"

    def collect_args(self) -> dict:
        return dict(nullable = self.nullable, precision = self.precision)

class Bool(_BaseDtype):
    def __str__(self):
        return f"Bool(null={self.nullable})"

class String(_BaseDtype):
    def __str__(self):
        return f"String(null={self.nullable})"

class DateT(_BaseDtype):
    def __str__(self):
        return f"DateT(null={self.nullable})"

class DateTimeT(_BaseDtype):
    def __init__(self, nullable: bool = True, precision:float = 0, tz: str = None):
        super().__init__(nullable)
        self.precision = precision
        self.tz = str(tz)  # whether timezone aware

    def __str__(self):
        return f"DateTimeT(null={self.nullable}, precision = {self.precision}, tz={self.tz})"

    def collect_args(self) -> dict:
        return dict(nullable = self.nullable, precision = self.precision, tz = self.tz)

class _ComplexDtype(_Dtype):
    pass

class Array(_ComplexDtype):
    def __init__(self, element_dtype: _Dtype):
        self.element_dtype = element_dtype

    def __str__(self):
        return f"Array<{self.element_dtype}>"

    @classmethod
    def from_args(cls, args: dict):
        # de-serialize
        element_config = args['element_dtype']
        return cls(element_dtype = _Dtype.from_config(element_config))

    def collect_args(self) -> dict:
        return dict(element_dtype = self.element_dtype.to_config())

class Map(_ComplexDtype):
    def __init__(self, key_dtype: _BaseDtype, value_dtype: _Dtype):
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

    def __str__(self):
        return f"Map<{self.key_dtype},{self.value_dtype}>"

    @classmethod
    def from_args(cls, args: dict):
        # de-serialize
        key_config = args['key_dtype']
        value_config = args['value_dtype']

        return cls(
            key_dtype = _Dtype.from_config(key_config),
            value_dtype = _Dtype.from_config(value_config)
        )

    def collect_args(self) -> dict:
        return dict(key_dtype = self.key_dtype.to_config(), value_dtype = self.value_dtype.to_config())

##### language specific interface

def check_auto_cast_precision(precision: int, supported_precision: list, force_cap: bool = False):
    if precision in supported_precision:
        # pass check
        return precision
    # if not, will cast to a bigger precision
    idx = np.searchsorted(supported_precision, precision, side='left')
    if idx >= len(supported_precision):
        if force_cap:
            logging.warning(f"Max precision exceeded, will force precision cap to {max(supported_precision)}")
            return max(supported_precision)
        raise TypeError(
            f"Precision Overflow, max support precision: {max(supported_precision)}, provided precision: {precision}")
    return supported_precision[idx]

class _DtypeBase:
    @classmethod
    def translate(cls, d: _Dtype) -> str:
        # auto detect what type is
        cls_name = d.__class__.__name__
        func = getattr(cls, f"to{cls_name}")
        return func(d)

    @classmethod
    def toInteger(cls, d: Integer) -> str:
        raise NotImplementedError

    @classmethod
    def toDecimal(cls, d: Decimal) -> str:
        raise NotImplementedError

    @classmethod
    def toFloat(cls, d: Float) -> str:
        raise NotImplementedError

    @classmethod
    def toBool(cls, d: Bool) -> str:
        raise NotImplementedError

    @classmethod
    def toString(cls, d: String) -> str:
        raise NotImplementedError

    @classmethod
    def toDateT(cls, d: DateT) -> str:
        raise NotImplementedError

    @classmethod
    def toDateTimeT(cls, d: DateTimeT) -> str:
        raise NotImplementedError

    @classmethod
    def toArray(cls, d: Array) -> str:
        raise NotImplementedError

    @classmethod
    def toMap(cls, d: Map) -> str:
        raise NotImplementedError

class DtypePandas(_DtypeBase):
    @classmethod
    def toInteger(cls, d: Integer) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[8, 16, 32, 64]
        )
        if d.nullable:
            # will use pandas extended type
            base = f"Int{precision}"
            if not d.signed:
                base = "U" + base
        else:
            # will use numpy type
            base = f"int{precision}"
            if not d.signed:
                base = "u" + base

        return base

    @classmethod
    def toDecimal(cls, d: Decimal) -> str:
        return "float64"

    @classmethod
    def toFloat(cls, d: Float) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[32, 64]
        )
        return f"float{precision}"

    @classmethod
    def toBool(cls, d: Bool) -> str:
        if d.nullable:
            return "boolean" # pandas extended type - nullable
        return "bool" # pandas original type

    @classmethod
    def toString(cls, d: String) -> str:
        return "string"

    @classmethod
    def toDateT(cls, d: DateT) -> str:
        # no date/datetime diff
        return "datetime64[ns]"

    @classmethod
    def toDateTimeT(cls, d: DateTimeT) -> str:
        if d.tz is not None:
            return f"datetime64[ns, {d.tz}]" # timezone aware type
        return "datetime64[ns]"

class DtypeSparkSQL(_DtypeBase):
    @classmethod
    def toInteger(cls, d: Integer) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[8, 16, 32, 64]
        )
        return {
            8: 'Byte',
            16: 'Short',
            32: 'Integer',
            64: 'Long'
        }.get(precision)

    @classmethod
    def toDecimal(cls, d: Decimal) -> str:
        return f"Decimal({d.precision},{d.scale})"

    @classmethod
    def toFloat(cls, d: Float) -> str:
        precision = check_auto_cast_precision(
            d.precision,
            supported_precision=[32, 64]
        )
        return {
            32: 'Float',
            64: 'Double'
        }.get(precision)

    @classmethod
    def toBool(cls, d: Bool) -> str:
        return "Boolean"

    @classmethod
    def toString(cls, d: String) -> str:
        return "String"

    @classmethod
    def toDateT(cls, d: DateT) -> str:
        return "Date"

    @classmethod
    def toDateTimeT(cls, d: DateTimeT) -> str:
        return "TimeStamp"

    @classmethod
    def toArray(cls, d: Array) -> str:
        e = d.element_dtype
        cls_name = e.__class__.__name__
        func = getattr(DtypeSparkSQL, f"to{cls_name}")
        base = func(e)
        return f"Array<{base}>"

    @classmethod
    def toMap(cls, d: Map) -> str:
        k = d.key_dtype
        v = d.value_dtype
        k_cls = k.__class__.__name__
        v_cls = v.__class__.__name__
        func_k = getattr(DtypeSparkSQL, f"to{k_cls}")
        func_v = getattr(DtypeSparkSQL, f"to{v_cls}")
        base_k = func_k(k)
        base_v = func_v(v)
        return f"Map<{base_k},{base_v}>"


######### Schema handling -- batch conversion

class Schema(OrderedDict):
    def __init__(self, schema_mapper: Dict[str, _Dtype]):
        """create Schema object from predefined schema

        :param schema_mapper: {feature_name, feature_type} type should use dtyper._Dtype
        """
        super().__init__(**schema_mapper)

    def to_js(self) -> OrderedDict:
        r = OrderedDict()
        for col, dtype in self.items():
            r[col] = dtype.to_config()
        return r
    
    def to_yaml(self, filename: str):
        with open(filename, 'w') as obj:
            yaml.dump(
                self.to_js(), 
                obj,
                default_flow_style=False, 
                sort_keys=False
            )

    @classmethod
    def from_js(cls, js: OrderedDict):
        """create Schema object from dictionary (json format)

        :param schema:
        :return: Schema()
        """
        s = Schema({})
        for col, config in js.items():
            s[col] = _Dtype.from_config(config)
        return s
    
    @classmethod
    def from_yaml(cls, filename: str):
        with open(filename, 'r') as obj:
            tt = yaml.load(obj, Loader=yaml.Loader)
            schemas = cls.from_js(tt)
        return schemas



if __name__ == '__main__':
    i = Integer(nullable=True, precision=64, signed=False)
    print(i)
    print(DtypeSparkSQL.toInteger(i), DtypeSparkSQL.translate(i))

    f = Array(Integer(nullable=True, precision=64, signed=False))
    print(f)
    print(DtypeSparkSQL.toArray(f), DtypeSparkSQL.translate(f))

    m = Map(
        String(nullable=False),
        Integer(nullable=True, precision=64, signed=False),
    )
    print(m)

    print(i.to_config())
    print(m.to_config())
    print(Array.from_config(f.to_config()))
    print(Map.from_config(m.to_config()))