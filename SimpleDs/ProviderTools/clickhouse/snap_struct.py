
from datetime import date
import logging
from typing import List
import pandas as pd
import pyspark
from CommonTools.sparker import SparkConnector
from CommonTools.utils import dt2str, str2dt
from CommonTools.SnapStructure.structure import SnapshotDataManagerBase
from ProviderTools.clickhouse.dbapi import ClickHouse, ClickhouseCRUD


class SnapshotDataManagerCHSQL(SnapshotDataManagerBase):
    """files are saved as clickhouse tables under each schema.table
    """

    @classmethod
    def setup(cls, ch_conf: ClickHouse, spark_connector: SparkConnector = None):
        super(SnapshotDataManagerCHSQL, cls).setup(spark_connector = spark_connector)
        cls.ch_conf = ch_conf
        cls.ch = ClickhouseCRUD(ch_conf)

    def __init__(self, schema:str, table:str, snap_dt_key: str):
        """database management interface

        :param schema: schema , e.g., RAW, PROCESSED
        :param table:  table under each schema, e.g., CLNT_GENERAL, BDA_GENERAL
        :param snap_dt_key: snap date column name for all tables
        """
        super().__init__(schema = schema, table = table)
        self.snap_dt_key = snap_dt_key

    def is_exist(self) -> bool:
        return self.ch.is_exist(schema=self.schema, table=self.table)

    def init_table(self, primary_keys:List[str], col_schemas: pd.DataFrame, table_note:str = None, overwrite:bool = False):
        if self.is_exist():
            if overwrite:
                self.ch.drop_table(schema=self.schema, table=self.table)
            else:
                logging.warning(f"{self.schema}.{self.table} already exists, will do nothing. set overwrite to True if you wish to reset table")
                return
        
        self.ch.create_schema(
            schema = self.schema
        )
        self.ch.create_table(
            schema = self.schema,
            table = self.table,
            col_schemas = col_schemas,
            engine = "MergeTree",
            partition_keys = [self.snap_dt_key],
            primary_keys = primary_keys,
            order_keys = primary_keys,
            table_note = table_note
        )

    def get_schema(self) -> pd.Series:
        """Check dtypes of the given schema/dataset

        :return:
        """
        schema = self.ch.detect_table_schema(self.schema, self.table)
        return pd.Series(schema['type'].tolist(), index = schema['name'].tolist(), name = 'dtype')

    def count(self, snap_dt: date) -> int:
        sql = "SELECT COUNT() FROM {schema:Identifier}.{table:Identifier} WHERE {partition_key:Identifier} = {snap_dt:Date32}"
        args = dict(schema=self.schema, table=self.table, partition_key = self.snap_dt_key, snap_dt=snap_dt)
        return self.ch.client.command(cmd = sql, parameters = args)

    def save_qry(self, query: str, snap_dt: date, overwrite: bool = False, keep_view:bool = True, **kws):
        """save the query into the table using "insert into schema.table select ... " clause

        :param query: the sql query for table to be inserted
        :param snap_dt: the snap date partition to be inserted, use to check existence
        :param overwrite: whether to overwrite the table if exists
        :param keep_view: the process will create temp view first, whether to keep that temp view
        :return:
        """
        # first detect whether snap date already exists
        clear = self.pre_save_check(snap_dt = snap_dt, overwrite = overwrite)
        if clear is False:
            return

        # first create a view of that table which can be inspected the column dtypes and names
        dt_str = dt2str(snap_dt, format = "%Y%m")
        view_name = f"{self.table}_V_{dt_str}"
        self.ch.create_view(schema = self.schema, view = view_name, query = query)
        qry_cols = self.ch.detect_table_schema(schema = self.schema, table = view_name)['name'].tolist()

        # insert into the table
        self.ch.insert_from_select(
            schema = self.schema,
            table = self.table,
            query = query,
            qry_cols = qry_cols # infered from view
        )

        # delete the view if required
        if not keep_view:
            self.ch.drop_view(schema = self.schema, view = view_name)

        logging.info(f"Successfully saved to {self.schema}.{self.table}@{snap_dt}")

    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        self.ch.client.insert_df(
            table = self.table,
            database = self.schema,
            df = df,
            **kws
        )

    def ingest_from_file(self, file_path:str, snap_dt: date, header:bool = True, overwrite:bool = True):
        if overwrite:
            self.delete(snap_dt = snap_dt)
        else:
            num_records = self.count(snap_dt)
            if num_records > 0:
                logging.warning(f"{num_records} records found for {self.schema}.{self.table}@{snap_dt}, will do nothing")
                return

        self.ch.ingest_file(
            schema = self.schema,
            table = self.table,
            file_path = file_path,
            header = header,
            use_cli = False
        )

    def get_existing_snap_dts(self) -> List[date]:
        sql = f"""
        select distinct {self.snap_dt_key}
        from {self.schema}.{self.table}
        order by {self.snap_dt_key}
        """
        existing_snaps = self.ch.client.query_df(sql)
        if len(existing_snaps) == 0:
            return []
        return list(str2dt(dt.date()) for dt in existing_snaps[self.snap_dt_key].dt.to_pydatetime())

    def read(self, snap_dt: date, **kws) -> pd.DataFrame:
        """Read as pandas dataframe (one snapshot date) data

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
        return self.ch.client.query_df(sql)

    def reads(self, snap_dts: List[date], **kws) -> pd.DataFrame:
        """reads as pandas dataframe (vertically concat of several given snap dates data)

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
        return self.ch.client.query_df(sql)


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
            df = self.sc.query_db(self.ch_conf, sql)
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
            df = self.sc.query_db(self.ch_conf, sql)
            df.createOrReplaceTempView(f"{self.table}")
            return df
        else:
            ValueError("No Spark Connector Specified, please call .setup() to bind a spark connector")


    def delete(self, snap_dt: date):
        """Delete a snap shot dataframe

        :param snap_dt: which snap date to delete
        :return:
        """
        self.ch.delete_data(schema=self.schema, table=self.table, where_clause = f"{self.snap_dt_key} = '{snap_dt}'")

    def drop(self):
        """drop the whole table

        :return:
        """
        self.ch.drop_table(schema=self.schema, table=self.table)

    def duplicate(self, dst_schema: str, dst_table: str):
        sql = """insert into {dst_schema:Identifier}.{dst_table:Identifier} select * from {src_schema:Identifier}.{src_table:Identifier}"""
        args = dict(dst_schema = dst_schema, dst_table = dst_table, src_schema = self.schema, src_table = self.table)
        return self.ch.client.command(cmd = sql, parameters = args)

    def disk_space(self, snap_dt, unit='MB') -> float:
        """get the size of the snap date file (pandas) or folder (pyspark partitions)

        :param snap_dt:
        :param unit: {KB, MB, GB}
        """
        sql = """
        select 
            sum(rows) as rows,
            sum(bytes_on_disk) as bytes_on_disk
        from system.parts
        where 
            active
            and database = %(schema)s
            and table = %(table)s
            and partition = %(snap_dt)s
        """
        args = dict(schema = self.schema, table =self.table, snap_dt = snap_dt)
        d = self.ch.client.query(query=sql, parameters=args).first_item

        size_bytes, rows = d['bytes_on_disk'], d['rows']
        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size

    def is_valid(self, snap_dt, rows_threshold: int = 0) -> bool:
        """Check if the file is valid"""
        sql = f"""
        select count() as size from {self.schema}.{self.table} where {self.snap_dt_key} = '{snap_dt}'
        """
        rows = self.ch.client.query(query = sql).first_item['size']
        return rows > rows_threshold