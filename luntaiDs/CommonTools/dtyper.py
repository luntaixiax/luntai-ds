import logging
from dataclasses import dataclass, asdict
from dacite import from_dict
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Any, Optional
import ibis
from ibis import schema
from ibis.expr.schema import Schema
from ibis.expr.datatypes import DataType

def extract_dtype(dtype: str, args: dict) -> DataType:
    if dtype == 'Array':
        value_type = args.pop('value_type')
        value_dtype = extract_dtype(value_type['dtype'], value_type['args'])
        return ibis.expr.datatypes.Array(
            value_type = value_dtype,
            **args
        )
    elif dtype == 'Map':
        key_type = args.pop('key_type')
        key_dtype = extract_dtype(key_type['dtype'], key_type['args'])
        value_type = args.pop('value_type')
        value_dtype = extract_dtype(value_type['dtype'], value_type['args'])
        return ibis.expr.datatypes.Map(
            key_type = key_dtype,
            value_type = value_dtype,
            **args
        )
    elif dtype == 'Struct':
        fields = args.pop("fields")
        fields_dtype = dict()
        for k, v in fields.items():
            v_dtype = extract_dtype(v['dtype'], v['args'])
            fields_dtype[k] = v_dtype
        return ibis.expr.datatypes.Struct(
            fields = fields_dtype,
            **args
        )
    else:
        return getattr(ibis.expr.datatypes, dtype)(**args)

@dataclass
class DSchemaField:
    """data schema field structure

    :param dtype: a string representing the basic type, support Array, Map, Struct
        from from https://ibis-project.org/reference/datatypes
    :param args: the argument for each schema fields, e.g., nullable: true
    :param primary_key: whether this field is part of primary key
    :param descr: a description
    :param extra_kws: extra keyword argument may be useful
    
    example settings:
        "CUST_ID" : {
            "dtype":"Int64",
            "args":{
                "nullable": false
            },
            "descr" : "customer id",
            "primary_key" : true
        },
        "SNAP_DT" : {
            "dtype":"Date",
            "args":{
                "nullable": false
            },
            "descr" : "observation snapshot date",
            "primary_key" : true
        },
        "NAME" : {
            "dtype":"String",
            "args":{
                "nullable": false
            },
            "descr" : "first and last name",
            "primary_key" : false
        },            
        "SAMPLE_ARRAY":{
            "dtype":"Array",
            "args":{
                "value_type":{
                    "dtype":"String",
                    "args":{
                        "nullable":false
                    }
                },
                "nullable":false
            },
            "descr" : "sample array",
        },
        "SAMPLE_MAP":{
            "dtype":"Map",
            "args":{
                "key_type":{
                    "dtype":"String",
                    "args":{
                        "nullable":false
                    }
                },
                "value_type":{
                    "dtype":"Float64",
                    "args":{
                        "nullable":false
                    }
                },
                "nullable":false
            },
            "descr" : "sample map",
        },
        "SAMPLE_NESTED":{
            "dtype":"Array",
            "args":{
                "value_type":{
                    "dtype":"Map",
                    "args":{
                        "key_type":{
                            "dtype":"String",
                            "args":{
                                "nullable":false
                            }
                        },
                        "value_type":{
                            "dtype":"Int64",
                            "args":{
                                "nullable":true
                            }
                        },
                        "nullable":false
                    }
                },
                "nullable":false
            },
            "descr" : "sample nested",
        },
        "SAMPLE_STRUCT":{
            "dtype":"Struct",
            "args":{
                "fields":{
                    "id":{
                        "dtype":"Int8",
                        "args":{
                            "nullable":false
                        }
                    },
                    "trans_dt":{
                        "dtype":"Timestamp",
                        "args":{
                            "timezone":"UTC",
                            "nullable":false
                        }
                    },
                    "amount":{
                    "dtype":"Decimal",
                        "args":{
                            "precision":10,
                            "scale":2,
                            "nullable":false
                        }
                    }
                },
                "nullable":false
            },
            "descr" : "sample struct",
        }
    """
    dtype: str
    args: Dict[str, Any]
    primary_key: bool = False
    cluster_key: bool = False
    partition_key: bool = False
    descr: Optional[str] = None
    extra_kws: Optional[Dict[str, Any]] = None
    
    @property
    def ibis_dtype(self) -> DataType:
        return extract_dtype(dtype = self.dtype, args = self.args)


class DSchema(OrderedDict):
    def __init__(self, dschema_fields: Dict[str, DSchemaField]):
        """create DSchema object from predefined dschema fields

        :param dschema_fields: {feature_name, feature_type} type should use DSchemaField
        """
        super().__init__(**dschema_fields)
        
    @property
    def ibis_schema(self) -> Schema:
        sh = []
        for col, schema_field in self.items():
            sh.append(
                (col, schema_field.ibis_dtype)
            )
            
        return schema(pairs = sh)
    
    @property
    def primary_keys(self) -> List[str]:
        # return primary keys
        return [k for k, v in self.items() if v.primary_key]
    
    @property
    def cluster_keys(self) -> List[str]:
        # return primary keys
        return [k for k, v in self.items() if v.cluster_key]
    
    @property
    def partition_keys(self) -> List[str]:
        # return primary keys
        return [k for k, v in self.items() if v.partition_key]
    
    @property
    def descrs(self) -> Dict[str, str]:
        # extract any column descriptions
        return {k: v.descr for k, v in self.items() if v.descr}
            
    @classmethod
    def from_js(cls, js: dict):
        """create DSchema object from dictionary (json format)

        :param js: the dictionary of configs, example see DSchemaField doc
        :return: DSchema
        """
        s = DSchema({})
        for col, config in js.items():
            s[col] = from_dict(
                data_class = DSchemaField,
                data = config
            )
        return s
    
    def to_js(self) -> OrderedDict:
        r = OrderedDict()
        for col, schema_field in self.items():
            r[col] = asdict(schema_field)
        return r
    
    

if __name__ == '__main__':
    pass