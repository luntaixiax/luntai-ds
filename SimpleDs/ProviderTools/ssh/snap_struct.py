from __future__ import annotations
import os
from datetime import date
from typing import List
import pandas as pd
from tqdm import tqdm
from CommonTools.utils import str2dt
from CommonTools.SnapStructure.tools import match_date_from_str
from CommonTools.SnapStructure.structure import SnapshotDataManagerBase
from CommonTools.sparker import SparkConnector
from ProviderTools.ssh.sftp import SFTP



class SnapshotDataManagerSFTP(SnapshotDataManagerBase):
    """files are saved as parquet snapshots under each schema.table
        file naming convention: dir/tablename_YYYY-MM-DD.parquet
    """
    
    @classmethod
    def setup(cls, root_dir: str, sftp: SFTP, spark_connector: SparkConnector = None, default_engine:str = 'pandas'):
        super(SnapshotDataManagerSFTP, cls).setup(
            spark_connector = spark_connector,
            default_engine = default_engine
        )
        cls.ROOT_DIR = root_dir
        cls.default_engine = default_engine
        cls.sftp = sftp
        
    def __init__(self, schema:str, table:str):
        """virtual database management interface

        :param schema: the virtual schema , e.g., RAW, PROCESSED
        :param table:  the virtual table under each schema
        """
        super().__init__(schema = schema, table = table)
        dir = os.path.join(self.ROOT_DIR, schema, table)
        self.dir = dir  # the root path of the table under each schema
        self.init_table()

    def init_table(self):
        self.sftp.mkdir(path = os.path.join(self.ROOT_DIR, self.schema), mode = 511, ignore_existing = True)
        self.sftp.mkdir(path = self.dir, mode = 511, ignore_existing = True)
        
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
        with self.sftp.getFileHandler(filepath, mode = 'w') as fp:
            df.to_parquet(fp, **kws)
         
    
    def get_existing_snap_dts(self) -> List[date]:
        file_list = self.sftp.ls(
            remoteFolderPath = self.dir,
            pattern = "*.parquet",
            verbose = False
        )
        existing_snaps = [
            str2dt(match_date_from_str(f))
            for f in file_list
        ]
        existing_snaps.sort()
        return existing_snaps
    
    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_default_file_path(snap_dt=snap_dt)
        with self.sftp.getFileHandler(filepath) as fp:
            df = pd.read_parquet(fp, **kws)
        return df
    
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
    
    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        filepath = self.get_default_file_path(snap_dt)
        self.sftp.delete(filepath)
        
    def drop(self):
        """drop the whole table

        :return:
        """
        self.sftp.delete_folder(self.dir)
        
    
    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerSFTP:
        new = SnapshotDataManagerSFTP(schema = dst_schema, table = dst_table)
        for snap_dt in tqdm(self.get_existing_snap_dts()):
            from_filename = self.get_default_file_path(snap_dt=snap_dt)
            to_filename = new.get_default_file_path(snap_dt=snap_dt)
            self.sftp.copy(from_filename, to_filename)
        return new
    
    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date file (pandas) or folder (pyspark partitions)

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        f = self.get_default_file_path(snap_dt)
        size_bytes = self.sftp.size(f)

        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size