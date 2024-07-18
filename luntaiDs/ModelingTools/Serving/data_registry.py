from pathlib import Path
import pyarrow.parquet as pq
from typing import List, Dict, Union
from fsspec import AbstractFileSystem
import ibis
import pandas as pd

from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.utils import save_ibis_to_parquet_on_fs, save_pandas_to_parquet_on_fs
from luntaiDs.CommonTools.warehouse import BaseWarehouseHandler

class _BaseModelDataRegistry:

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        raise NotImplementedError("")

    def register(self, data_id: str, train_ds: Union[ibis.expr.types.Table | pd.DataFrame], 
                 test_ds: Union[ibis.expr.types.Table | pd.DataFrame], replace: bool = False):
        """register table in ibis format

        :param str data_id: data id
        :param Union[ibis.expr.types.Table | pd.DataFrame] train_ds: training dataset
        :param Union[ibis.expr.types.Table | pd.DataFrame] test_ds: testing dataset
        :param bool replace: whether to replace existing dataset, defaults to False
        """
        raise NotImplementedError("")

    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        raise NotImplementedError("")

    def remove(self, data_id: str):
        """remove dataset from registry

        :param str data_id: data id to be removed
        """
        raise NotImplementedError("")


class ModelDataRegistryFileSystem(_BaseModelDataRegistry):
    def __init__(self, fs: AbstractFileSystem, data_root: str):
        """modeling data registry implemented for fsspec compatible file system
        file structure:
        
            data_root
            - data_id
                - train.parquet
                - test.parquet

        :param AbstractFileSystem fs: the fsspec compatible filesystem
        :param str data_root: root path, if on object storage, 
            the full path including buckets
        """
        self._fs = fs
        self._data_root = data_root
        self._fs.makedirs(
            path = data_root,
            exist_ok = True
        )

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        try:
            files = self._fs.listdir(path = self._data_root, detail = False)
            ids = [Path(file).name for file in files]
        except FileNotFoundError as e:
            ids = []
        return ids

    def get_train_path(self, data_id: str) -> str:
        return (Path(self._data_root) / data_id / f"train_{data_id}.parquet").as_posix()

    def get_test_path(self, data_id: str) -> str:
        return (Path(self._data_root) / data_id / f"test_{data_id}.parquet").as_posix()

    def register(self, data_id: str, train_ds: Union[ibis.expr.types.Table | pd.DataFrame], 
                 test_ds: Union[ibis.expr.types.Table | pd.DataFrame], replace: bool = False):
        """register table in ibis format

        :param str data_id: data id
        :param Union[ibis.expr.types.Table | pd.DataFrame] train_ds: training dataset
        :param Union[ibis.expr.types.Table | pd.DataFrame] test_ds: testing dataset
        :param bool replace: whether to replace existing dataset, defaults to False
        """
        existing_ids = self.get_existing_ids()
        if not replace and data_id in existing_ids:
            raise ValueError(f"data id {data_id} already exist, pls use another id")
        
        self._fs.makedirs(
            path = (Path(self._data_root) / data_id).as_posix(),
            exist_ok = True
        )
        
        if isinstance(train_ds, pd.DataFrame) and isinstance(train_ds, pd.DataFrame):
            save_pandas_to_parquet_on_fs(
                df = train_ds,
                fs = self._fs,
                filepath = self.get_train_path(data_id)
            )
            save_ibis_to_parquet_on_fs(
                df = test_ds,
                fs = self._fs,
                filepath = self.get_test_path(data_id)
            )
        elif isinstance(train_ds, ibis.expr.types.Table) and isinstance(train_ds, ibis.expr.types.Table):
            save_ibis_to_parquet_on_fs(
                df = train_ds,
                fs = self._fs,
                filepath = self.get_train_path(data_id)
            )
            save_ibis_to_parquet_on_fs(
                df = test_ds,
                fs = self._fs,
                filepath = self.get_test_path(data_id)
            )
        else:
            raise TypeError(f"train_ds/test_ds must be either ibis or pandas dataframe")


    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        assert data_id in self.get_existing_ids(), f"data id {data_id} does not exist"
        
        train_filepath = self.get_train_path(data_id)
        test_filepath = self.get_test_path(data_id)
        
        try:
            # attempt to read from duckdb directly if possible
            con = ibis.duckdb.connect()
            con.register_filesystem(self._fs)
            train_ds = ibis.read_parquet(
                self._fs.unstrip_protocol(
                    name = train_filepath
                )
            )
            test_ds = ibis.read_parquet(
                self._fs.unstrip_protocol(
                    name = test_filepath
                )
            )
        except Exception as e:
            # will use fs native open method to read
            with self._fs.open(path = train_filepath) as obj:
                train_arr = pq.read_table(
                    source = obj
                )
                train_ds = ibis.memtable(train_arr)
            with self._fs.open(path = test_filepath) as obj:
                test_arr = pq.read_table(
                    source = obj
                )
                test_ds = ibis.memtable(test_arr)
            
        if target_col:
            X_train = train_ds.drop(target_col)
            X_test = test_ds.drop(target_col)
            y_train = train_ds[target_col]
            y_test = test_ds[target_col]
            return X_train, y_train, X_test, y_test
        else:
            return train_ds, test_ds

    def remove(self, data_id: str):
        """remove dataset from registry

        :param str data_id: data id to be removed
        """
        dpath = (Path(self._data_root) / data_id).as_posix()
        self._fs.rm(
            path = dpath,
            recursive = True
        )


class _ModelDataRegistryWarehouse(_BaseModelDataRegistry):
    DATA_ID_COL = "DATA_ID_"
    TRAIN_TEST_IND_COL = "IS_TRAIN_"
    
    def __init__(self, handler: BaseWarehouseHandler, schema: str, table: str):
        """initialize

        :param BaseWarehouseHandler handler: base warehouse handler
        :param str schema: schema for the table storing modeling data
        :param str table: table storing modeling data
        """
        self.handler = handler
        self.schema = schema
        self.table = table
        
    def init(self, col_schemas: DSchema, **kws):
        """warehouse need create and initialize schema/table structure at beginning

        :param DSchema col_schemas: column schema object
        """
        # create the schema first
        self.handler.create_schema(schema = self.schema)
        
        # validate if table already exists
        if self.handler.is_exist(schema = self.schema, table = self.table):
            # if exists, check if schema changes
            existing_schema = self.handler.get_dtypes(
                schema = self.schema, 
                table = self.table
            )
            current_schema = col_schemas.ibis_schema
            if not current_schema.equals(existing_schema):
                msg = f"""
                {self.schema}.{self.table} table schema changed:
                Current in DB:
                {existing_schema}
                Now:
                {current_schema}
                
                Recreate table schema or fix it
                """
                raise TypeError(msg)
            return
        
        # create the table
        self.handler.create_table(
            schema = self.schema,
            table = self.table,
            col_schemas = col_schemas,
            **kws
        )
    
    def get_table(self) -> ibis.expr.types.Table:
        """get underlying training table using ibis

        :return ibis.expr.types.Table: ibis table
        """
        return self.handler.get_table(schema = self.schema, table = self.table)

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        return (
            self.get_table()
            .select(self.DATA_ID_COL)
            .distinct()
            .to_pandas()
            [self.DATA_ID_COL]
            .tolist()
        )
            

    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        table = self.get_table()
        train_ds = (
            table
            .filter((table[self.DATA_ID_COL] == data_id) & (table[self.TRAIN_TEST_IND_COL] == True))
            .drop(self.DATA_ID_COL, self.TRAIN_TEST_IND_COL)
        )
        test_ds = (
            table
            .filter((table[self.DATA_ID_COL] == data_id) & (table[self.TRAIN_TEST_IND_COL] == False))
            .drop(self.DATA_ID_COL, self.TRAIN_TEST_IND_COL)
        )
        if target_col:
            X_train = train_ds.drop(target_col)
            X_test = test_ds.drop(target_col)
            y_train = train_ds[target_col]
            y_test = test_ds[target_col]
            return X_train, y_train, X_test, y_test
        else:
            return train_ds, test_ds