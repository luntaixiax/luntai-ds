from __future__ import annotations
from typing import List
from pathlib import Path
import pandas as pd
import fsspec
import logging
import ibis
from datetime import date
from luntaiDs.CommonTools.SnapStructure.tools import match_date_from_str
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.utils import save_ibis_to_parquet_on_fs, str2dt

    
class SnapshotDataManagerBase:
    @classmethod
    def setup(cls,):
        pass

    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema , e.g., RAW, PROCESSED
        :param table:  the virtual table under each schema, e.g., CLNT_GENERAL, BDA_GENERAL
        """
        self.schema = schema
        self.table = table

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}; table={self.table})"
    
    def exist(self) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created

        :return bool: whether the table and schema exists and ready
        """
        raise NotImplementedError("")
    
    def init_table(self, *args, **kws):
        """initialize schema/table

        """
        raise NotImplementedError("")
    
    def get_schema(self) -> ibis.expr.schema.Schema:
        """Check dtypes of the given schema/dataset

        :return: the ibis schema of the given table
        """
        return self.read_random_snap().schema()

    
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
                logging.warning(f"Presave check failed, {num_records} existing records found for "
                                f"{self.schema}.{self.table}@{snap_dt} while overwrite is set to False")
                return False
            return True
    
    def _save(self, df: ibis.expr.types.Table, snap_dt: date, **kws):
        """The pure logic to save ibis dataframe to the system, without handling existing record problem

        :param ibis.expr.types.Table df: the ibis dataframe
        :param date snap_dt: the snap date to save
        """
        raise NotImplementedError("")
    
    def save(self, df: ibis.expr.types.Table, snap_dt: date, overwrite: bool = False, **kws):
        """save pandas dataframe to the system

        :param ibis.expr.types.Table df: the ibis dataframe
        :param date snap_dt: the snap date to save
        :param overwrite snap_dt: whether to overwrite the existing record if any
        :raises NotImplementedError: _description_
        """
        if not self.pre_save_check(snap_dt = snap_dt, overwrite = overwrite):
            return
        
        logging.info(f"Saving dataframe, schema {df.schema()} into {self.schema}.{self.table}")
        self._save(df = df, snap_dt = snap_dt, **kws)
        logging.info(f"Successfully saved {df.count()} records to {self.schema}.{self.table}@{snap_dt}")
        
    def get_existing_snap_dts(self) -> List[date]:
        """get existing snap dates as a list

        :return List[date]: list of snap dates
        """
        raise NotImplementedError("")
    
    def count(self, snap_dt: date) -> int:
        """count the sample size for given snap date

        :param date snap_dt: _description_
        :return int: # of rows of the dataframe
        """
        if snap_dt not in self.get_existing_snap_dts():
            return 0
        df = self.read(snap_dt=snap_dt)
        return df.count().to_pandas() # convert to scalar
    
    def read(self, snap_dt: date, **kws) -> ibis.expr.types.Table:
        """Read as ibis dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        raise NotImplementedError("")
    
    def reads(self, snap_dts: List[date], **kws) -> ibis.expr.types.Table:
        """reads as ibis dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        raise NotImplementedError("")

    def read_random_snap(self) -> ibis.expr.types.Table:
        """Select a random snap from the schema.table

        :return:
        """
        from random import choice

        existing_snaps = self.get_existing_snap_dts()
        random_snap = choice(existing_snaps)
        return self.read(random_snap)
    
    def read_pd(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        raise NotImplementedError("")
    
    def reads_pd(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        raise NotImplementedError("")
    
    def read_random_pd_snap(self) -> pd.DataFrame:
        """Select a random snap from the schema.table

        :return:
        """
        from random import choice

        existing_snaps = self.get_existing_snap_dts()
        random_snap = choice(existing_snaps)
        return self.read_pd(random_snap)
    
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

    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerBase:
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
    
    
class SnapshotDataManagerFileSystem(SnapshotDataManagerBase):
    """files are saved as parquet snapshots under each schema.table
        file naming convention: dir/tablename_YYYY-MM-DD.parquet
        using fsspec engine for different implementation of FS
            - local
            - s3/minio
            - gcs
            - azure blob storage
            - http FS
    """
    @classmethod
    def setup(cls, fs: fsspec.AbstractFileSystem, root_dir: str):
        super(SnapshotDataManagerFileSystem, cls).setup()
        cls.ROOT_DIR = root_dir
        #cls._fs: fsspec.AbstractFileSystem = ArrowFSWrapper(fs)
        cls._fs = fs
        # duckdb engine to read table
        cls._ibis_con = ibis.duckdb.connect()
        cls._ibis_con.register_filesystem(fs)
        
    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema
        :param table:  the virtual table under each schema
        """
        super().__init__(schema = schema, table = table)
        dir = Path(self.ROOT_DIR) / schema / table
        self.dir: str = dir.as_posix()  # the root path of the table under each schema
        self.init_table()
        
    def init_table(self):
        """initialize schema/table
        
        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.archive.AbstractArchiveFileSystem.info
        """
        self._fs.makedirs(
            path = self.dir,
            exist_ok = True
        )
        
    def exist(self) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created

        :return bool: whether the table and schema exists and ready
        """
        return self._fs.exists(path = self.dir)
    
    def get_filename(self, snap_dt: date) -> str:
        """get parquet file name

        :param date snap_dt: the snap date of the file
        :return str: the file name ending with .parquet, excluding filepath
        """
        return f"{self.table}_{snap_dt}.parquet"
    
    def get_file_path(self, snap_dt: date) -> str:
        """the absolute path to the parquet file

        :param date snap_dt: the snap date of the file
        :return str: absolute file path
        """
        return (Path(self.dir) / self.get_filename(snap_dt)).as_posix()
    
    def get_file_path_with_protocol(self, snap_dt: date) -> str:
        """the absolute path to the parquet file, including FS protocol

        :param date snap_dt: the snap date of the file
        :return str: e.g., in s3fs, return s3://bucket/file/path/to/your.parquet
        """
        return self._fs.unstrip_protocol(
            name = self.get_file_path(snap_dt)
        )
    
    def _save(self, df: ibis.expr.types.Table, snap_dt: date, **kws):
        """The pure logic to save ibis dataframe to the system, without handling existing record problem

        :param ibis.expr.types.Table df: the ibis dataframe
        :param date snap_dt: the snap date to save
        """
        save_ibis_to_parquet_on_fs(
            df = df,
            fs = self._fs,
            filepath = self.get_file_path(snap_dt)
        )
                    
    def get_existing_snap_dts(self) -> List[date]:
        """get existing snap dates as a list

        :return List[date]: list of snap dates
        """
        existing_files: List[str] = self._fs.ls(
            path = self.dir, 
            detail = False
        )
        filenames = [Path(file).name for file in existing_files]
        existing_snaps = list(set(str2dt(match_date_from_str(f)) for f in filenames))
        existing_snaps.sort()
        return existing_snaps
                        
    def read(self, snap_dt: date, **kws) -> ibis.expr.types.Table:
        """Read as ibis dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_file_path_with_protocol(snap_dt=snap_dt)
        
        return self._ibis_con.read_parquet(filepath, **kws)
        
    
    def reads(self, snap_dts: List[date], **kws) -> ibis.expr.types.Table:
        """reads as ibis dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        filepaths = (self.get_file_path_with_protocol(snap_dt) for snap_dt in snap_dts)
        return self._ibis_con.read_parquet(filepaths, **kws)
    
    def read_pd(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_file_path(snap_dt=snap_dt)
        return pd.read_parquet(
            path = filepath,
            engine = 'pyarrow',
            filesystem = self._fs,
            **kws
        )
    
    def reads_pd(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
                # do the checks to ensure successful concatenation
        li =  []
        for snap_dt in snap_dts:
            df_partition = self.read_pd(snap_dt=snap_dt, **kws)
            li.append(df_partition)

        return pd.concat(li, axis = 0, ignore_index = True)
    
    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        filepath = self.get_file_path(snap_dt=snap_dt)
        if self._fs.exists(filepath):
            self._fs.rm_file(path = filepath)

    def drop(self):
        """drop the whole table

        :return:
        """
        if self.exist():
            self._fs.rm(
                path = self.dir,
                recursive = True,
                maxdepth = None
            )

    def migrate(self, dst_schema: str, dst_table: str):
        """move from the existing schema.table to new one, the existing one will be deleted

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        new = self.duplicate(dst_schema, dst_table)
        self.drop()
        return new

    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerFileSystem:
        """duplicate the existing schema.table to new one, the existing one will be kept

        reference here: https://filesystem-spec.readthedocs.io/en/latest/copying.html
        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        new = type(self)(
            schema = dst_schema, 
            table = dst_table
        )
        # copy folder to new location
        self._fs.cp(
            path1 = str(Path(self.dir) / "_")[:-1],
            path2 = str(Path(new.dir) / "_")[:-1],
            recursive = True,
            maxdepth = None
        )
        

    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date on disk

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        filepath = self.get_file_path(snap_dt=snap_dt)
        size_bytes = self._fs.du(
            path = filepath,
            total = True,
            maxdepth = None
        )
        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size
    
        
class SnapshotDataManagerWarehouseMixin(SnapshotDataManagerBase):
    """Mixin class for warehouse implementation (partial)
    """
    def __init__(self, schema:str, table:str, snap_dt_key: str):
        """database management interface

        :param schema: schema
        :param table:  table under each schema
        :param snap_dt_key: snap date column name for all tables
        """
        super().__init__(schema = schema, table = table)
        self.snap_dt_key = snap_dt_key
        
    def exist(self) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created

        :return bool: whether the table and schema exists and ready
        """
        return self.is_exist(schema=self.schema, table=self.table)
    
    def init_table(self, col_schemas: DSchema, overwrite:bool = False, **settings):
        """initialize/create table in the underlying data warehouse system

        :param DSchema col_schemas: data column schema
        :param bool overwrite: whether to drop table if exists, defaults to False
        """

        if self.exist():
            if overwrite:
                self.drop()
            else:
                logging.warning(f"{self.schema}.{self.table} already exists, will do nothing." 
                                "set overwrite to True if you wish to reset table")
                return
        
        # create schema
        self.create_schema(schema = self.schema)
        # create table
        self.create_table(
            schema = self.schema,
            table = self.table,
            col_schemas = col_schemas,
            **settings
        )
    
    def get_schema(self) -> ibis.expr.schema.Schema:
        """Check dtypes of the given schema/dataset

        :return: the ibis schema of the given table
        """
        return self.get_dtypes(
            schema = self.schema,
            table = self.table
        )
    
    
    def get_existing_snap_dts(self) -> List[date]:
        """get existing snap dates as a list

        :return List[date]: list of snap dates
        """
        if not self.exist():
            return []
        existing_snaps = (
            self.get_table(
                schema = self.schema,
                table = self.table
            ).select(self.snap_dt_key)
            .distinct()
            .order_by(self.snap_dt_key)
            .to_pandas()
        )
        if len(existing_snaps) == 0:
            return []
        return list(
            str2dt(dt.date()) 
            for dt in pd.to_datetime(existing_snaps[self.snap_dt_key]).dt.to_pydatetime()
        )
    
    def read(self, snap_dt: date, **kws) -> ibis.expr.types.Table:
        """Read as ibis dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        table = self.get_table(
            schema = self.schema,
            table = self.table
        )
        df = table.filter(table[self.snap_dt_key] == snap_dt)
        if 'columns' in kws:
            df = df.select(*kws['columns'])
        return df
    
    def reads(self, snap_dts: List[date], **kws) -> ibis.expr.types.Table:
        """reads as ibis dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        table = self.get_table(
            schema = self.schema,
            table = self.table
        )
        df = table.filter(table[self.snap_dt_key].isin(snap_dts))
        if 'columns' in kws:
            df = df.select(*kws['columns'])
        return df
    
    def read_pd(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        return self.read(snap_dt=snap_dt, **kws).to_pandas()
    
    def reads_pd(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        return self.reads(snap_dts=snap_dts, **kws).to_pandas()

    def drop(self):
        """drop the whole table

        :return:
        """
        self.delete_table(
            schema = self.schema,
            table = self.table,
        )