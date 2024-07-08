import ibis
import mlflow

def _ibis_dtype_2_mlflow_dtype_mapping(ibis_dtype: ibis.expr.datatypes.DataType
        ) -> mlflow.types.schema.DataType | mlflow.types.schema.Array | mlflow.types.schema.Map:
    """convert ibis data dtype to mlflow data dtype

    :param ibis.expr.datatypes.DataType ibis_dtype: ibis dtype, see 
        https://ibis-project.org/reference/datatypes#parameters
    :return mlflow.types.schema.DataType | mlflow.types.schema.Array | mlflow.types.schema.Map:
        mlflow support either single value dtype, array or map type, see
        https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.DataType
    """
    if ibis_dtype.is_binary():
        return mlflow.types.schema.DataType.binary
    elif ibis_dtype.is_boolean():
        return mlflow.types.schema.DataType.boolean
    elif ibis_dtype.is_date() or ibis_dtype.is_timestamp():
        return mlflow.types.schema.DataType.datetime
    elif ibis_dtype.is_float64() or ibis_dtype.is_decimal():
        return mlflow.types.schema.DataType.double
    elif ibis_dtype.is_float32() or ibis_dtype.is_float16():
        return mlflow.types.schema.DataType.float
    elif ibis_dtype.is_integer(): 
        if ibis_dtype.is_int8() or ibis_dtype.is_int16() or ibis_dtype.is_int32() \
            or ibis_dtype.is_uint8() or ibis_dtype.is_uint16() or ibis_dtype.is_uint32():
            return mlflow.types.schema.DataType.integer
        else:
            return mlflow.types.schema.DataType.long
    elif ibis_dtype.is_array():
        return mlflow.types.schema.Array(
            dtype = _ibis_dtype_2_mlflow_dtype_mapping(ibis_dtype.value_type)
        )
    elif ibis_dtype.is_map():
        if not ibis_dtype.key_type.is_string():
            raise TypeError("Only support string key dtype for Map mlflow schema type")
        return mlflow.types.schema.Map(
            value_type = _ibis_dtype_2_mlflow_dtype_mapping(ibis_dtype.value_type)
        )
    else:
        return mlflow.types.schema.DataType.string


def ibis_dtype_2_mlflow_dtype(colname: str, ibis_dtype: ibis.expr.datatypes.DataType
        ) -> mlflow.types.schema.ColSpec:
    """convert from ibis dtype to mlflow ColSpec (similar to column in database)

    :param str colname: column name
    :param ibis.expr.datatypes.DataType ibis_dtype: ibis data dtype
    :return mlflow.types.schema.ColSpec: mlflow ColSpec
    """
    mlflow_dtype = _ibis_dtype_2_mlflow_dtype_mapping(ibis_dtype)
    
    return mlflow.types.schema.ColSpec(
        name = colname,
        required = not ibis_dtype.nullable,
        type = mlflow_dtype
    )
    
def ibis_schema_2_mlflow_schema(ibis_schema: ibis.expr.schema.Schema
        ) -> mlflow.types.schema.Schema:
    """convert ibis table schema to mlflow table schema

    :param ibis.expr.schema.Schema ibis_schema: ibis table schema
    :return mlflow.types.schema.Schema: mlflow table schema
    """
    mlflow_dtypes = []
    for col, dtype in ibis_schema.items():
        mlflow_dtypes.append(ibis_dtype_2_mlflow_dtype(
            colname = col, 
            ibis_dtype = dtype
        ))
    return mlflow.types.schema.Schema(inputs = mlflow_dtypes)