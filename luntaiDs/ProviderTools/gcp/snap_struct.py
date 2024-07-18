
from __future__ import annotations
from datetime import date
import logging
from typing import List
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
import ibis
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerWarehouseMixin
from luntaiDs.ProviderTools.gcp.dbapi import BigQuery, WarehouseHandlerBQSQL

class SnapshotDataManagerBQSQL(SnapshotDataManagerWarehouseMixin, WarehouseHandlerBQSQL):
    """files are saved as bigquery tables under each schema.table
    """
    @classmethod
    def setup(cls, db_conf: BigQuery, **settings):
        super(SnapshotDataManagerBQSQL, cls).setup()
        cls.connect(db_conf=db_conf, **settings)
        
    def save_qry(self, query: str, snap_dt: date, overwrite: bool = False):
        """save the query into the table using "insert into schema.table select ... " clause

        :param query: the sql query for table to be inserted
        :param snap_dt: the snap date partition to be inserted, use to check existence
        :param overwrite: whether to overwrite the table if exists

        :return:
        """
        # first detect whether snap date already exists
        clear = self.pre_save_check(snap_dt = snap_dt, overwrite = overwrite)
        if clear is False:
            return

        # first create a view of that table which can be inspected the column dtypes and names
        qry_cols = self.query(query).columns

        # insert into the table
        # clickhouse insert need column order correct
        sql = f"""
        INSERT {self.schema}.{self.table} ({','.join(qry_cols)})
        {query}
        """
        logging.info(f"Inserting into {self.schema}.{self.table}@{snap_dt} using query:\n{sql}")
        self.execute(sql)

        logging.info(f"Successfully saved to {self.schema}.{self.table}@{snap_dt}")
        
    def _save(self, df: ibis.expr.types.Table, snap_dt: date, **kws):
        """The pure logic to save ibis dataframe to the system, without handling existing record problem

        :param ibis.expr.types.Table df: the ibis dataframe
        :param date snap_dt: the snap date to save
        """
        self.save_ibis(
            df = df,
            schema = self.schema,
            table = self.table,
            **kws
        )
        
    def ingest_from_file(self, snap_dt: date, file_path: str, 
                        fs: AbstractFileSystem = LocalFileSystem(), 
                        overwrite:bool = True, **settings):
        
        if overwrite:
            self.delete(snap_dt = snap_dt)
        else:
            num_records = self.count(snap_dt)
            if num_records > 0:
                logging.warning(f"{num_records} records found for "
                                f"{self.schema}.{self.table}@{snap_dt}, will do nothing")
                return
        
        with fs.open(file_path, 'rb') as obj:
            if file_path.endswith("parquet"):
                df_arr = pq.read_table(
                    source = obj,
                    **settings
                )
            elif file_path.endswith("csv"):
                df_arr = pcsv.read_csv(
                    input_file = obj,
                    **settings
                )
            else:
                raise TypeError("File not supported")
        
        self._ops.insert(
            table_name = self.table,
            obj = df_arr,
            database = self.schema,
            overwrite = False
        )
        
        
    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        if self.exist():
            sql = f"""
            DELETE {self.schema}.{self.table}
            WHERE {self.snap_dt_key} = '{snap_dt}'
            """
            logging.info(f"Deleting table {self.schema}.{self.table} using query:\n{sql}")
            self.execute(sql = sql)
        
    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerBQSQL:
        sql = f"""
        CREATE TABLE {dst_schema}.{dst_table} 
        COPY {self.schema}.{self.table}"""
        self.execute(sql = sql)
        new = SnapshotDataManagerBQSQL(
            schema = dst_schema,
            table = dst_table,
            snap_dt_key = self.snap_dt_key
        )
        return new
    
    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date file (pandas) or folder (pyspark partitions)

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        if not self.exist():
            return 0
        
        tb = self._ops.table(f"{self.schema}.__TABLES__")
        # we only get total size, no partition size on date
        summs = (
            tb
            .filter(tb['table_id'] == self.table)
            .agg(
                size = tb['size_bytes'].sum(),
                rows = tb['row_count'].sum()
            )
            .to_pandas()
        )
        size_bytes = summs.loc[0, 'size']
        rows = summs.loc[0, 'rows']
        row_snap = self.count(snap_dt)
        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes * row_snap / (rows * 1024 ** scale)
        return size