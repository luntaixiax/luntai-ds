import os
from datetime import date
import logging
from typing import List
import pandas as pd
import pyspark
from tqdm.auto import tqdm

from CommonTools.sparker import SparkConnector
from CommonTools.utils import dt2str, str2dt
from CommonTools.SnapStructure.structure import SnapshotDataManagerBase
from CommonTools.SnapStructure.tools import match_date_from_str
from ProviderTools.aws.s3 import S3Accessor


class SnapshotDataManagerS3(SnapshotDataManagerBase):
    """files are saved as s3 objs under each schema.table
    """

    @classmethod
    def setup(cls, bucket: str,  root_dir: str, s3a: S3Accessor, spark_connector: SparkConnector = None, default_engine:str = 'pandas'):
        super(SnapshotDataManagerS3, cls).setup(
            spark_connector = spark_connector,
            default_engine = default_engine
        )
        cls.bucket = bucket
        cls.ROOT_DIR = root_dir if not root_dir.startswith("/") else root_dir[1:]
        cls.s3a = s3a
        cls.s3a.enter_bucket(bucket_name=bucket)
    
    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema , e.g., RAW, PROCESSED
        :param table:  the virtual table under each schema, e.g., CLNT_GENERAL, BDA_GENERAL
        """
        super().__init__(schema = schema, table = table)
        dir = os.path.join(self.ROOT_DIR, schema, table)
        self.dir = dir  # the root path of the table under each schema

    def get_filename(self, snap_dt: date) -> str:
        return f"{self.table}_{snap_dt}.parquet"

    def get_default_file_path(self, snap_dt: date) -> str:
        return os.path.join(self.dir, self.get_filename(snap_dt))
    
    def get_file_path_with_s3_prefix(self, snap_dt: date) -> str:
        return f"s3a://{self.bucket}/{self.get_default_file_path(snap_dt=snap_dt)}"

    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        self.s3a.save_parquet(
            df = df,
            remote_path = self.get_default_file_path(snap_dt=snap_dt),
            **kws
        )
    
    def _save_partitions(self, df: pyspark.sql.DataFrame, snap_dt: date, **kws):
        """the pure logic to save pyspark dataframe to the system, without handling existing record problem

        :param pyspark.sql.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError("")
    
    def get_existing_snap_dts(self) -> List[date]:
        df = self.s3a.list_objs(remote_path = self.dir)
        existing_snaps = list(set(map(
            str2dt,
            (
                df
                .loc[
                    (df['Key'].str.startswith(pat = f"{self.dir}/"))
                    & (df['Key'].str.endswith(pat = ".parquet")), 
                    'Key']
                .str.replace(self.dir, "")
                .apply(match_date_from_str)
                .tolist()
            )
        )))
        existing_snaps.sort()
        return existing_snaps
    
    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_default_file_path(snap_dt=snap_dt)
        return self.s3a.read_parquet(remote_path = filepath, **kws)
    

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
        filepath = self.get_file_path_with_s3_prefix(snap_dt)
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
        file_paths = [self.get_file_path_with_s3_prefix(snap_dt) for snap_dt in snap_dts]
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
        """Delete a snap shot dataframe, could be partitions

        :param snap_dt: which snap date to delete
        :return:
        """
        filepath = self.get_default_file_path(snap_dt=snap_dt)
        if self.s3a.list_objs(filepath) is None or len(self.s3a.list_objs(filepath)) == 1:
            # it is not folder
            self.s3a.delete_obj(remote_path = filepath)
        else:
            # it is a folder, i.e., partitions
            self.s3a.delete_folder(remote_folder_path = filepath)

    def drop(self):
        """drop the whole table

        :return:
        """
        self.s3a.delete_folder(remote_folder_path = self.dir)

    def duplicate(self, dst_schema: str, dst_table: str):
        """duplicate the existing schema.table to new one, the existing one will be kept

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        new = SnapshotDataManagerS3(schema = dst_schema, table = dst_table)
        for snap_dt in tqdm(self.get_existing_snap_dts()):
            from_filename = self.get_default_file_path(snap_dt=snap_dt)
            to_filename = new.get_default_file_path(snap_dt=snap_dt)
            self.s3a.copy_obj(
                from_path = from_filename,
                to_path = to_filename
            )
        

    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date on disk

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        #key = self.s3a.get_obj(remote_path=self.get_default_file_path(snap_dt=snap_dt))
        #size_bytes = key['ContentLength'] # in kb
        size_bytes = self.s3a.list_objs(remote_path=self.get_default_file_path(snap_dt=snap_dt))['Size'].sum()
        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size
    
    def is_valid(self, snap_dt, size_threshold: float = 0.5) -> bool:
        """Check if the file is valid
        
        """
        return self.disk_space(snap_dt, unit = 'MB') > size_threshold