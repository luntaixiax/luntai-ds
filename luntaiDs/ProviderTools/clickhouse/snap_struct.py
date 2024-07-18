
from __future__ import annotations
from datetime import date
import logging
from typing import List
import ibis
from ibis import _
from fsspec.implementations.local import LocalFileSystem
from fsspec import AbstractFileSystem
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerWarehouseMixin
from luntaiDs.ProviderTools.clickhouse.dbapi import ClickHouse, WarehouseHandlerCHSQL

class SnapshotDataManagerCHSQL(SnapshotDataManagerWarehouseMixin, WarehouseHandlerCHSQL):
    """files are saved as clickhouse tables under each schema.table
    """
    @classmethod
    def setup(cls, db_conf: ClickHouse, **settings):
        super(SnapshotDataManagerCHSQL, cls).setup()
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
        INSERT INTO {self.schema}.{self.table} ({','.join(qry_cols)})
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
                        header:bool = True, overwrite:bool = True, 
                        column_names: List[str] = None, compression: str = None, **settings):
        """ingest from a parquet file from any fsspec compatible filesystem

        :param date snap_dt: snap date for the file to ingest
        :param str file_path: the absolute filepath on the given filesystem
        :param AbstractFileSystem fs: fsspec compatible FS, defaults to LocalFileSystem()
        :param bool header: where have a header or not, defaults to True
        :param bool overwrite: whether to overwrite the existing data, defaults to True
        :param List[str] column_names: used in clickhouse raw insert, defaults to None
        :param str compression: compression format, defaults to None
        """
        if overwrite:
            self.delete(snap_dt = snap_dt)
        else:
            num_records = self.count(snap_dt)
            if num_records > 0:
                logging.warning(f"{num_records} records found for "
                                f"{self.schema}.{self.table}@{snap_dt}, will do nothing")
                return
            
        # extract file format
        if file_path.endswith("parquet"):
            fmt = "Parquet"
        elif file_path.endswith("csv"):
            fmt = "CSVWithNames" if header else "CSV"
        else:
            raise TypeError("File not supported")
        
        from clickhouse_connect.driver.query import quote_identifier

        if not self.schema and self.table[0] not in ('`', "'") and self.table.find('.') > 0:
            full_table = self.table
        elif self.schema:
            full_table = f'{quote_identifier(self.schema)}.{quote_identifier(self.table)}'
        else:
            full_table = quote_identifier(self.table)
        if not fmt:
            fmt = 'CSVWithNames'
        if compression is None:
            if file_path.endswith('.gzip') or file_path.endswith('.gz'):
                compression = 'gzip'
        with fs.open(file_path, 'rb') as obj:
            self._ops.con.raw_insert(
                full_table,
                column_names=column_names,
                insert_block=obj,
                fmt=fmt,
                settings=settings,
                compression=compression
            )
        logging.info(f"Successfully ingested file {file_path} to {self.schema}.{self.table}")

        
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
        """duplicate the existing schema.table to new one, the existing one will be kept

        :param dst_schema: destination schema
        :param dst_table: destination table
        :return:
        """
        args = dict(
            dst_schema = dst_schema, 
            dst_table = dst_table, 
            src_schema = self.schema, 
            src_table = self.table
        )
        # create table structure
        sql = f"""
        create table if not exists {dst_schema}.{dst_table}
        AS {self.schema}.{self.table}"""
        self.execute(sql = sql)
        
        # insert from select
        sql = """
        insert into {dst_schema:Identifier}.{dst_table:Identifier} 
        select * from {src_schema:Identifier}.{src_table:Identifier}"""
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