
from __future__ import annotations
from datetime import date
import logging
from typing import List
import pandas as pd
import pyspark
from clickhouse_connect.driver.tools import insert_file
from luntaiDs.CommonTools.sparker import SparkConnector
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerWarehouseMixin
from luntaiDs.ProviderTools.clickhouse.dbapi import ClickHouse, WarehouseHandlerCHSQL

class SnapshotDataManagerCHSQL(SnapshotDataManagerWarehouseMixin, WarehouseHandlerCHSQL):
    """files are saved as clickhouse tables under each schema.table
    """
    @classmethod
    def setup(cls, db_conf: ClickHouse, spark_connector: SparkConnector = None, **settings):
        super(SnapshotDataManagerCHSQL, cls).setup(spark_connector = spark_connector)
        cls.connect(db_conf=db_conf, settings=settings)
        
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
        INSERT INTO {self.schema}.{self.table} ({','.join(qry_cols)})
        {query}
        """
        logging.info(f"Inserting into {self.schema}.{self.table}@{snap_dt} using query:\n{sql}")
        self.execute(sql)

        logging.info(f"Successfully saved to {self.schema}.{self.table}@{snap_dt}")
        
    def _save(self, df: pd.DataFrame, snap_dt: date, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        self.save_pandas(
            df = df,
            schema = self.schema,
            table = self.table,
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
            
        # extract file format
        if file_path.endswith("parquet"):
            fmt = "Parquet"
        elif file_path.endswith("csv"):
            fmt = "CSVWithNames" if header else "CSV"
        else:
            raise TypeError("File not supported")
        
        insert_file(
            client = self._ops.con,
            table = self.table,
            database = self.schema,
            file_path = file_path,
            fmt = fmt,
        )
        logging.info(f"Successfully ingested file {file_path} to {self.schema}.{self.table}")
    
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
        if self.exist():
            sql = f"""
            ALTER TABLE {self.schema}.{self.table} DELETE
            WHERE {self.snap_dt_key} = '{snap_dt}'
            """
            logging.info(f"Deleting table {self.schema}.{self.table} using query:\n{sql}")
            self.execute(sql = sql)
        
    def duplicate(self, dst_schema: str, dst_table: str) -> SnapshotDataManagerCHSQL:
        sql = """
        insert into {dst_schema:Identifier}.{dst_table:Identifier} 
        select * from {src_schema:Identifier}.{src_table:Identifier}"""
        args = dict(
            dst_schema = dst_schema, 
            dst_table = dst_table, 
            src_schema = self.schema, 
            src_table = self.table
        )
        self.execute(sql = sql, parameters = args)
        new = SnapshotDataManagerCHSQL(
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
        if not self.exist():
            return 0
        
        args = dict(schema = self.schema, table =self.table, snap_dt = snap_dt)
        d = self._ops.con.query(query=sql, parameters=args).first_item

        size_bytes, rows = d['bytes_on_disk'], d['rows']
        scale = {'KB': 1, 'MB': 2, 'GB': 3}.get(unit, 0)
        size = size_bytes / (1024 ** scale)
        return size