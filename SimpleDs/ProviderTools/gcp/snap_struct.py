
from __future__ import annotations
from datetime import date
import logging
from typing import List
import ibis
from ibis.expr.schema import Schema
import pandas as pd
import pyspark
from google.oauth2 import service_account
from CommonTools.sparker import SparkConnector
from CommonTools.utils import dt2str, str2dt
from CommonTools.dtyper import DSchema
from CommonTools.SnapStructure.structure import SnapshotDataManagerBase
from ProviderTools.gcp.dbapi import BigQuery

class SnapshotDataManagerBQSQL(SnapshotDataManagerBase):
    """files are saved as bigquery tables under each schema.table
    """
    @classmethod
    def setup(cls, db_conf: BigQuery, spark_connector: SparkConnector = None, **settings):
        super(SnapshotDataManagerBQSQL, cls).setup(spark_connector = spark_connector)
        cls._db_conf = db_conf
        if db_conf.credentials_path:
            creds = service_account.Credentials.from_service_account_file(db_conf.credentials_path)
        else:
            creds = None
        cls._ops = ibis.bigquery.connect(
            project_id=db_conf.project_id,
            dataset_id=db_conf.db,
            credentials = creds if creds else None,
            **settings
        )
        
    def __init__(self, schema:str, table:str, snap_dt_key: str):
        """database management interface

        :param schema: schema
        :param table:  table under each schema
        :param snap_dt_key: snap date column name for all tables
        """
        super().__init__(schema = schema, table = table)
        self.snap_dt_key = snap_dt_key
        
    def is_exist(self) -> bool:
        return self.table in self._ops.list_tables(like = self.table, schema = self.schema)

    def init_table(self, col_schemas: DSchema, overwrite:bool = False, **settings):
        """initialize/create table in the underlying data warehouse system

        :param DSchema col_schemas: data column schema
        :param bool overwrite: whether to drop table if exists, defaults to False
        """

        if self.is_exist():
            if overwrite:
                self.drop()
            else:
                logging.warning(f"{self.schema}.{self.table} already exists, will do nothing." 
                                "set overwrite to True if you wish to reset table")
                return
        
        # create schema
        self._ops.create_schema(
            name = self.schema,
            force = True
        )
        # create table
        primary_keys = col_schemas.primary_keys
        partition_keys = col_schemas.partition_keys
        if self.snap_dt_key not in partition_keys:
            partition_keys.append(self.snap_dt_key)
        cluster_keys = col_schemas.cluster_keys
        logging.info(f"Creating table {self.schema}.{self.table} using schema:\n{col_schemas.ibis_schema}")
        logging.info(f"primary keys = {primary_keys}, partition_keys = {partition_keys}")
        # create table
        self._ops.create_table(
            name = self.table,
            schema = col_schemas.ibis_schema,
            database = self.schema,
            overwrite = overwrite,
            partition_by = partition_keys,
            cluster_by = cluster_keys,
            options = settings
        )
        # add column descriptions
        descrs = col_schemas.descrs
        for col, descr in descrs.items():
            try:
                sql = f"""
                ALTER TABLE {self.schema}.{self.table} 
                ALTER COLUMN {col} 
                SET OPTIONS (
                    description="{descr}"
                )"""
                self._ops.con.command(sql)
            except:
                pass
        
    def get_schema(self) -> Schema:
        """Check dtypes of the given schema/dataset

        :return:
        """
        return self._ops.get_schema(
            name = self.table,
            schema = self.schema
        )
        
    def count(self, snap_dt: date) -> int:
        table = self._ops.table(
            name = self.table,
            schema = self.schema
        )
        return table.count(
            where = (table[self.snap_dt_key] == snap_dt)
        ).to_pandas() # convert to a scalar number
        
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
        qry_cols = self._ops.sql(query).columns

        # insert into the table
        # clickhouse insert need column order correct
        sql = f"""
        INSERT {self.schema}.{self.table} ({','.join(qry_cols)})
        {query}
        """
        logging.info(f"Inserting into {self.schema}.{self.table}@{snap_dt} using query:\n{sql}")
        self._ops.raw_sql(query = sql)

        logging.info(f"Successfully saved to {self.schema}.{self.table}@{snap_dt}")
        
    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        self._ops.client.load_table_from_dataframe(
            dataframe = df,
            destination = f"{self.schema}.{self.table}",
            **kws
        )
        
    def get_existing_snap_dts(self) -> List[date]:
        existing_snaps = (
            self._ops.table(
                name = self.table,
                schema = self.schema
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
        
    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        table = self._ops.table(
            name = self.table,
            schema = self.schema
        )
        df = table.filter(table[self.snap_dt_key] == snap_dt)
        if 'columns' in kws:
            df = df.select(*kws['columns'])
        return df.to_pandas()
    
    def reads(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        table = self._ops.table(
            name = self.table,
            schema = self.schema
        )
        df = table.filter(table[self.snap_dt_key].isin(snap_dts))
        if 'columns' in kws:
            df = df.select(*kws['columns'])
        return df.to_pandas()
    
    def load(self, snap_dt: date, **kws) -> pyspark.sql.DataFrame:
        """Read as spark dataframe (one snapshot date) data, and can also access from sc temporary view

        :param snap_dt: snap_dt to load
        :return:
        """
        if 'columns' in kws:
            cols = ','.join(kws['columns'])
        else:
            cols = '*'
        sql = f"""
        select {cols} from {self.schema}.{self.table} where {self.snap_dt_key} = '{snap_dt}'
        """
        if hasattr(self, "sc"):
            df = self.sc.query_db(self._db_conf, sql)
            df.createOrReplaceTempView(f"{self.table}")
            return df
        else:
            ValueError("No Spark Connector Specified, please call .setup() to bind a spark connector")

    def loads(self, snap_dts: List[date], **kws) -> pyspark.sql.DataFrame:
        """reads as pyspark dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        if 'columns' in kws:
            cols = ','.join(kws['columns'])
        else:
            cols = '*'
        snap_dt_range = ",".join(f"'{dt}'" for dt in snap_dts)
        sql = f"""
        select {cols} from {self.schema}.{self.table} where {self.snap_dt_key} in [{snap_dt_range}]
        """
        if hasattr(self, "sc"):
            df = self.sc.query_db(self._db_conf, sql)
            df.createOrReplaceTempView(f"{self.table}")
            return df
        else:
            ValueError("No Spark Connector Specified, please call .setup() to bind a spark connector")
        
    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        sql = f"""
        DELETE {self.schema}.{self.table}
        WHERE {self.snap_dt_key} = '{snap_dt}'
        """
        logging.info(f"Deleting table {self.schema}.{self.table} using query:\n{sql}")
        self._ops.raw_sql(query = sql)
        
    def drop(self):
        """drop the whole table

        :return:
        """
        self._ops.drop_table(
            name = self.table,
            schema = self.schema,
            force = True
        )
        
    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerBQSQL:
        sql = f"""
        CREATE TABLE {dst_schema}.{dst_table} 
        COPY {self.schema}.{self.table}"""
        self._ops.raw_sql(query = sql)
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