from __future__ import annotations
import os
import shutil
from datetime import date
import time
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import List
import logging
import pandas as pd
import pyspark
from tqdm.auto import tqdm

from CommonTools.SnapStructure.tools import get_file_list_pattern, match_date_from_str
from CommonTools.dbapi import baseDbInf, dbIO
from CommonTools.sparker import SparkConnector
from CommonTools.utils import str2dt

    
class SnapshotDataManagerBase:
    @classmethod
    def setup(cls, spark_connector: SparkConnector = None, default_engine:str = 'pandas'):
        if spark_connector:
            cls.sc = spark_connector
        cls.default_engine = default_engine

    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema , e.g., RAW, PROCESSED
        :param table:  the virtual table under each schema, e.g., CLNT_GENERAL, BDA_GENERAL
        """
        self.schema = schema
        self.table = table

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}; table={self.table})"
    
    def init_table(self, *args, **kws):
        """initialize schema/table

        :return:
        """
        raise NotImplementedError("")
    
    def get_schema(self) -> pd.Series:
        """Check dtypes of the given schema/dataset

        :return:
        """

        if self.default_engine == 'pandas':
            df_random = self.read_random_snap()
            return df_random.dtypes
        if self.default_engine == 'spark':
            df_random = self.load_random_snap()
            return pd.Series(dict(df_random.dtypes))
        else:
            raise ValueError("engine can only be spark or pandas")
    
    def pre_save_check(self, snap_dt: date, overwrite: bool = False) -> bool:
        """pre-check before inserting records, will delete existing records at snap_dt if overwrite is set to True

        :param snap_dt:
        :param overwrite: if False, will not insert record if exists; will delete otherwise
        :return: True = clear to go, False = exit in downstream
        """
        if overwrite:
            self.delete(snap_dt = snap_dt)
            return True
        else:
            num_records = self.count(snap_dt)
            if num_records > 0:
                logging.warning(f"Presave check failed, {num_records} existing records found for {self.schema}.{self.table}@{snap_dt} while overwrite is set to False")
                return False
            return True
    
    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError("")
    
    def save(self, df: pd.DataFrame, snap_dt: date, overwrite: bool = False, **kws):
        """save pandas dataframe to the system

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :param overwrite snap_dt: whether to overwrite the existing record if any
        :raises NotImplementedError: _description_
        """
        if not self.pre_save_check(snap_dt = snap_dt, overwrite = overwrite):
            return
        
        self._save(df = df, snap_dt = snap_dt, **kws)
        logging.info(f"Successfully saved {len(df)} records to {self.schema}.{self.table}@{snap_dt}")

    def _save_partitions(self, df: pyspark.sql.DataFrame, snap_dt: date, **kws):
        """the pure logic to save pyspark dataframe to the system, without handling existing record problem

        :param pyspark.sql.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError("")

    def save_partitions(self, df: pyspark.sql.DataFrame, snap_dt: date, overwrite: bool = False, **kws):
        """save pyspark dataframe to the system

        :param pyspark.sql.DataFrame df: _description_
        :param date snap_dt: _description_
        :param overwrite snap_dt: whether to overwrite the existing record if any
        :raises NotImplementedError: _description_
        """
        if not self.pre_save_check(snap_dt = snap_dt, overwrite = overwrite):
            return
        
        self._save_partitions(df = df, snap_dt = snap_dt, **kws)
        logging.info(f"Successfully saved {df.count()} records to {self.schema}.{self.table}@{snap_dt}")

    def etl_from_db(self, db_conf: baseDbInf, sql: str, snap_dt: date, engine: str = 'pandas'):
        """do ETL work, extracting data from database and save it to local dir

        :param db_conf: the database configuration object, specify which db and connection param
        :param sql: the sql to do the ETL
        :param snap_dt: the snap date to do the ETL, for storage purpose
        :param engine: the engine to query_extract {pandas, spark}
        :return:
        """
        if engine == "pandas":
            db_conn = dbIO(db_conf)
            df = db_conn.query_sql_df(sql)
            self.save(df, snap_dt)
        elif engine == "spark":
            # spark data query_extract
            if hasattr(self, "sc"):
                df = self.sc.query_db(db_conf, sql)
                self.save_partitions(df, snap_dt)
            else:
                raise ValueError("No spark session binded, please bind via .setup() method")
        else:
            raise ValueError("Engine can only be pandas or spark")
        
    def get_existing_snap_dts(self) -> List[date]:
        raise NotImplementedError("")
    
    def count(self, snap_dt: date) -> int:
        """count the sample size for given snap date

        :param date snap_dt: _description_
        :return int: # of rows of the dataframe
        """
        if snap_dt not in self.get_existing_snap_dts():
            return 0
        if self.default_engine == 'pandas':
            df = self.read(snap_dt=snap_dt)
            return len(df)
        if self.default_engine == 'spark':
            df = self.load(snap_dt=snap_dt)
            return df.count()
        else:
            raise ValueError("engine can only be spark or pandas")
    
    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        raise NotImplementedError("")
    
    def reads(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        raise NotImplementedError("")

    def read_random_snap(self) -> pd.DataFrame:
        """Select a random snap from the schema.table

        :return:
        """
        import random

        existing_snaps = self.get_existing_snap_dts()
        random_snap = random.choice(existing_snaps)
        return self.read(random_snap)
    
    def load(self, snap_dt: date, **kws) -> pyspark.sql.DataFrame:
        """Read as spark dataframe (one snapshot date) data, and can also access from sc temporary view

        :param snap_dt: snap_dt to load
        :return:
        """
        raise NotImplementedError("")

    def load_random_snap(self) -> pyspark.sql.DataFrame:
        """Select a random snap from the schema.table

        :return:
        """
        import random

        existing_snaps = self.get_existing_snap_dts()
        random_snap = random.choice(existing_snaps)
        return self.load(random_snap)
    
    def loads(self, snap_dts: List[date], **kws) -> pyspark.sql.DataFrame:
        """reads as pyspark dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        raise NotImplementedError("")
    
    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        raise NotImplementedError("")

    def drop(self):
        """drop the whole table

        :return:
        """
        raise NotImplementedError("")

    def migrate(self, dst_schema: str, dst_table: str):
        """move from the existing schema.table to new one, the existing one will be deleted

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        new = self.duplicate(dst_schema, dst_table)
        self.drop()
        return new

    def duplicate(self, dst_schema: str, dst_table: str):
        """duplicate the existing schema.table to new one, the existing one will be kept

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        raise NotImplementedError("")

    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date on disk

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        raise NotImplementedError("")
    
    def is_valid(self, snap_dt, rows_threshold: int = 100) -> bool:
        rows = self.count(snap_dt = snap_dt)
        return rows > rows_threshold
    


class SnapshotDataManagerLocalFS(SnapshotDataManagerBase):
    """files are saved as parquet snapshots under each schema.table
        file naming convention: dir/tablename_YYYY-MM-DD.parquet
    """

    @classmethod
    def setup(cls, root_dir: str, spark_connector: SparkConnector = None, default_engine:str = 'pandas'):
        super(SnapshotDataManagerLocalFS, cls).setup(
            spark_connector = spark_connector,
            default_engine = default_engine
        )
        cls.ROOT_DIR = root_dir
        cls.default_engine = default_engine
        
    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema , e.g., RAW, PROCESSED
        :param table:  the virtual table under each schema, e.g., CLNT_GENERAL, BDA_GENERAL
        """
        super().__init__(schema = schema, table = table)
        dir = os.path.join(self.ROOT_DIR, schema, table)
        self.dir = dir  # the root path of the table under each schema
        self.init_table()

    def init_table(self):
        # https://note.nkmk.me/en/python-os-mkdir-makedirs/
        os.makedirs(self.dir, exist_ok = True)

    def get_filename(self, snap_dt: date) -> str:
        return f"{self.table}_{snap_dt}.parquet"

    def get_default_file_path(self, snap_dt: date) -> str:
        return os.path.join(self.dir, self.get_filename(snap_dt))
    

    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        filepath = self.get_default_file_path(snap_dt)
        df.to_parquet(filepath, **kws)
        logging.info(f"Successfully saved file to {filepath}")

    def _save_partitions(self, df: pyspark.sql.DataFrame, snap_dt: date, **kws):
        """the pure logic to save pyspark dataframe to the system, without handling existing record problem

        :param pyspark.sql.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        filepath = self.get_default_file_path(snap_dt)
        self.sc.save_parquet(df, filepath, **kws)
        logging.info(f"Successfully saved file to {filepath}")

    def get_existing_snap_dts(self) -> List[date]:
        existing_files = get_file_list_pattern(
            self.dir,
            pattern = f"*{self.table}*",
            include_dir = False  # don't include directory path
        )

        existing_snaps = list(set(str2dt(match_date_from_str(f)) for f in existing_files))
        existing_snaps.sort()
        return existing_snaps

    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_default_file_path(snap_dt)
        return pd.read_parquet(filepath, **kws)

    def reads(self, snap_dts: List[date], partition_key: str = None, **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :param partition_key: when vertically combining, need partition key to tell apart, typically its something like SNAP_DT
                if not specified, will bypass checking step
        :return:
        """
        # do the checks to ensure successful concatenation
        li =  []
        columns = pd.Index([])
        for snap_dt in tqdm(snap_dts):
            df_partition = self.read(snap_dt=snap_dt, **kws)
            if partition_key is not None and partition_key not in df_partition.columns:
                raise ValueError(f"{self.schema}.{self.table}: Partition Key {partition_key} not found in {snap_dt}")
            if len(columns) > 0 and len(columns.union(df_partition.columns)) != len(columns):
                # if new column set has a different length than the previous one, means the columns are not same across all the table partitions
                raise ValueError(f"{self.schema}.{self.table}: Columns not consistent for snapshot {snap_dt}")
            li.append(df_partition)

        return pd.concat(li, axis = 0, ignore_index = True)


    def load(self, snap_dt: date, **kws) -> pyspark.sql.DataFrame:
        """Read as spark dataframe (one snapshot date) data, and can also access from sc temporary view

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_default_file_path(snap_dt)
        if hasattr(self, "sc"):
            df = self.sc.read_parquet(filepath, **kws)
            df.createOrReplaceTempView(f"{self.table}")
            return df
        else:
            ValueError("No Spark Connector Specified, please call .setup() to bind a spark connector")

    def loads(self, snap_dts: List[date], partition_key: str = None, **kws) -> pyspark.sql.DataFrame:
        """reads as pyspark dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :param partition_key: when vertically combining, need partition key to tell apart, typically its something like SNAP_DT
                if not specified, will bypass checking step
        :return:
        """
        file_paths = [self.get_default_file_path(snap_dt) for snap_dt in snap_dts]
        if hasattr(self, "sc"):
            df = self.sc.read_parquet(*file_paths, **kws) # spark will automatically stack all tables together
            # do the partition check
            if partition_key is not None and partition_key not in df.columns:
                raise ValueError(f"{self.schema}.{self.table}: Partition Key {partition_key} not found")

            # pick up random snapshot data to check the columns are consistent
            one_columns = self.load_random_snap().columns
            if len(one_columns) != len(pd.Index(one_columns).union(df.columns)):
                raise ValueError(f"{self.schema}.{self.table}: Columns not consistent in one of the snapshot data")

            df.createOrReplaceTempView(f"{self.table}")
            return df
        else:
            ValueError("No Spark Connector Specified, please call .setup() to bind a spark connector")

    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        filepath = self.get_default_file_path(snap_dt)
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                # parquet partitions
                shutil.rmtree(filepath, ignore_errors = True)
            else:
                os.remove(filepath)

    def drop(self):
        """drop the whole table

        :return:
        """
        shutil.rmtree(self.dir)

    def migrate(self, dst_schema: str, dst_table: str):
        """move from the existing schema.table to new one, the existing one will be deleted

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        new = self.duplicate(dst_schema, dst_table)
        self.drop()
        return new

    def duplicate(self, dst_schema: str, dst_table: str):
        new = SnapshotDataManagerLocalFS(schema = dst_schema, table = dst_table)
        copy_tree(self.dir, new.dir)
        return new

    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date file (pandas) or folder (pyspark partitions)

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        f = self.get_default_file_path(snap_dt)
        if os.path.isfile(f):
            size_bytes = os.path.getsize(f)
        else:
            size_bytes = sum(p.stat().st_size for p in Path(f).rglob('*'))

        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size

    def is_valid(self, snap_dt, size_threshold: float = 0.5) -> bool:
        """Check if the file is valid
        
        """
        f = self.get_default_file_path(snap_dt)
        if os.path.isdir(f):
            # if it is a folder (partition files), it should has a .__SUCCESS.crc file generated by pyspark
            return "._SUCCESS.crc" in os.listdir(f)
        else:
            # if a single file, the size must be exceed the valid threshold
            return self.disk_space(snap_dt, unit = 'MB') > size_threshold