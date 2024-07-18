import logging
import os
import subprocess
from typing import List, Tuple
import clickhouse_connect
from clickhouse_connect.driver.tools import insert_file
import pandas as pd
import ibis
import pyarrow as pa
from luntaiDs.CommonTools.dbapi import baseDbInf
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.warehouse import BaseWarehouseHandler

class ClickHouse(baseDbInf):
    def __init__(self, driver="clickhouse+native"):
        super().__init__(driver)

    def argTempStr(self):
        return "%s"

    def getJdbcUrl(self) -> str:
        return f"jdbc:clickhouse://{self.ip}:{self.port}/{self.db}"

    def getDriverClass(self) -> str:
        # https://repo1.maven.org/maven2/com/github/housepower/clickhouse-native-jdbc-shaded/2.6.5/clickhouse-native-jdbc-shaded-2.6.5.jar
        return "com.github.housepower.jdbc.ClickHouseDriver"

    def getConnStr(self) -> str:
        # only used for sqlalchemy driver type
        return f"{self.driver}://{self.username}:{self.password}@{self.ip}:{self.port}"

class WarehouseHandlerCHSQL(BaseWarehouseHandler):
    @classmethod
    def connect(cls, db_conf: ClickHouse, **settings):
        cls._db_conf = db_conf
        cls._ops = ibis.clickhouse.connect(
            user=db_conf.username,
            password=db_conf.password,
            host=db_conf.ip,
            port=db_conf.port,
            **settings
        )
        
    def is_exist(self, schema: str, table: str) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created
        
        :param str schema: schema/database
        :param str table: table name
        :return bool: whether the table and schema exists and ready
        """
        if schema not in self._ops.list_databases(like = schema):
            return False
        return table in self._ops.list_tables(like = table, database = schema)
    
    def get_table(self, schema: str, table: str) -> ibis.expr.types.Table:
        """get the ibis table
        
        :param str schema: schema/database
        :param str table: table name
        """
        return self._ops.table(name = table, database = schema)
    
    def create_schema(self, schema: str):
        """create database/schema
        
        :param str schema: schema/database
        """
        self._ops.create_database(
            name = schema,
            force = True
        )
        
    def create_table(self, schema: str, table: str, col_schemas: DSchema, **settings):
        """initialize/create table in the underlying data warehouse system

        :param str schema: schema/database
        :param str table: table name
        :param DSchema col_schemas: data column schema
        :param List[str] primary_keys: primary keys, defaults to None
        """
        # by default, use merge tree engine
        engine = settings.pop('engine', 'MergeTree')
        # create table
        primary_keys = col_schemas.primary_keys
        partition_keys = col_schemas.partition_keys
        logging.info(f"Creating table {schema}.{table} using schema:\n{col_schemas.ibis_schema}")
        logging.info(f"primary keys = {primary_keys}, partition_keys = {partition_keys}")
        self._ops.create_table(
            name = table,
            schema = col_schemas.ibis_schema,
            database = schema,
            engine = engine,
            order_by = primary_keys,
            partition_by = partition_keys,
            settings = settings
        )
        # add column descriptions
        descrs = col_schemas.descrs
        for col, descr in descrs.items():
            try:
                sql = f"""
                ALTER TABLE {schema}.{table} 
                COMMENT COLUMN IF EXISTS {col} '{descr}'"""
                self._ops.con.command(sql)
            except:
                pass
            
    def get_dtypes(self, schema: str, table: str) -> ibis.Schema:
        """Check dtypes of the given schema/dataset

        :return:
        """
        return self._ops.get_schema(
            table_name = table,
            database = schema
        )
        
    def query(self, sql: str, schema: str = None) -> ibis.expr.types.Table:
        """query using SQL

        :return: ibis dataframe
        """
        return self._ops.sql(sql)
    
    def execute(self, sql: str, **kws):
        """execute SQL for operations like insert/update/delete

        :return: ibis dataframe
        """
        self._ops.con.command(sql, **kws)
        
    def save_pandas(self, df: pd.DataFrame, schema: str, table: str, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param str schema: schema/database
        :param str table: table name
        """
        self._ops.insert(
            name = table,
            obj = df,
            database = schema,
            **kws
        )
    
    def save_ibis(self, df: ibis.expr.types.Table, schema: str, table: str, **kws):
        """The pure logic to save ibis dataframe to the system, without handling existing record problem

        :param ibis.expr.types.Table df: ibis table
        :param str schema: schema/database
        :param str table: table name
        """
        chunk_size = kws.get('chunk_size', 1048576)
        df_arr: pa.RecordBatchReader = df.to_pyarrow_batches(chunk_size = chunk_size)
        for df_batch in df_arr:
            df_: pa.Table = pa.Table.from_batches([df_batch], schema = df_batch.schema)
            self._ops.insert(
                name = table,
                obj = df_,
                database = schema,
                **kws
            )
        
        
    def delete_table(self, schema: str, table: str):
        """drop whole table
        
        :param str schema: schema/database
        :param str table: table name
        """
        self._ops.drop_table(
            name = table,
            database = schema,
            force = True
        )
    
    def truncate_table(self, schema: str, table: str):
        """Truncate (remove) data from table
        
        :param str schema: schema/database
        :param str table: table name
        """
        self._ops.truncate_table(
            name = table,
            database = schema
        )
        

"""Below are Legacy Implementation"""
class ClickhouseSchema:
    def __init__(self, dtype_dict: dict, default_dict: dict = None, note_dict: dict = None):
        self.dtype_dict = dtype_dict
        self.default_dict = default_dict if default_dict is not None else {}
        self.note_dict = note_dict if note_dict is not None else {}

    def to_dataframe(self) -> pd.DataFrame:
        s = pd.merge(
            pd.Series(self.dtype_dict, dtype='string').rename("dtype"),
            pd.Series(self.default_dict, dtype='string').rename("default"),
            how='left',
            right_index=True,
            left_index=True
        )
        s = pd.merge(
            s,
            pd.Series(self.note_dict, dtype='string').rename("note"),
            how='left',
            right_index=True,
            left_index=True
        )
        return s

    @classmethod
    def from_dataframe(cls, schema_df: pd.DataFrame):
        return cls(
            dtype_dict=schema_df['dtype'].to_dict(),
            default_dict=schema_df['default'].dropna().to_dict(),
            note_dict=schema_df['note'].dropna().to_dict(),
        )

    def to_sql(self) -> str:
        l = []
        for col, dtype in self.dtype_dict.items():
            base = f"{col} {dtype}"
            if self.default_dict.get(col) is not None:
                default_v = self.default_dict.get(col)
                base += f" DEFAULT {default_v}"
            if self.note_dict.get(col) is not None:
                base += f" COMMENT '{self.note_dict.get(col)}'"

            l.append(base)

        return ",\n".join(l)


class ClickhouseCRUD:
    @classmethod
    def setup(cls, base_path: str):
        cls.base_path = base_path

    def __init__(self, ch_conf: ClickHouse):
        self.client = clickhouse_connect.get_client(
            host=ch_conf.ip,
            port=ch_conf.port,
            username=ch_conf.username,
            password=ch_conf.password
        )

    def is_exist(self, schema: str, table: str) -> bool:
        sql = "EXISTS {schema:Identifier}.{table:Identifier}"
        args = dict(schema = schema, table = table)
        r = self.client.command(cmd = sql, parameters = args) # will return single value for simple query
        return r

    def extract_file_suffix(self, file_path: str, header: bool = True) -> str:
        if file_path.endswith("parquet"):
            fmt = "Parquet"
        elif file_path.endswith("csv"):
            fmt = "CSVWithNames" if header else "CSV"
        else:
            raise TypeError("File not supported")
        return fmt
    
    def show_tables(self, schema: str) -> List[str]:
        sql = "SELECT DISTINCT name FROM system.tables WHERE database={schema:String} AND has_own_data = 1"
        args = dict(schema = schema)
        df = self.client.query_df(query = sql, parameters = args)
        return df['name'].tolist()

    def create_schema(self, schema: str):
        sql = "CREATE DATABASE IF NOT EXISTS {schema:Identifier}"
        args = dict(schema = schema)
        self.client.command(cmd = sql, parameters = args)
        logging.info(f"Successfully created schema {schema}")

    def detect_file_schema(self, file_path: str, header: bool = True) -> pd.DataFrame:
        fmt = self.extract_file_suffix(file_path=file_path, header=header)
        sql = "DESCRIBE TABLE file({file_path: String}, {fmt:Identifier})"
        args = dict(file_path = file_path, fmt = fmt)
        return self.client.query_df(query = sql, parameters = args)

    def detect_table_schema(self, schema: str, table: str) -> pd.DataFrame:
        sql = "DESCRIBE TABLE {schema:Identifier}.{table:Identifier}"
        args = dict(schema=schema, table=table)
        return self.client.query_df(query = sql, parameters = args)

    def mount_file_src(self, src_folder_path: str, dest_folder_name: str = None) -> str:
        if dest_folder_name is None:
            dest_folder_name = os.path.relpath(src_folder_path, os.path.dirname(src_folder_path))
        dest_path = os.path.join(self.base_path, "usr/bin/user_files", dest_folder_name)
        command = f"ln -s {src_folder_path} {dest_path}"
        proc = subprocess.Popen(command, shell=True, close_fds=True)
        return command

    def unmount_file_src(self, dest_folder_name: str) -> str:
        dest_path = os.path.join(self.base_path, "usr/bin/user_files", dest_folder_name)
        command = f"unlink {dest_path}"
        proc = subprocess.Popen(command, shell=True, close_fds=True)
        return command

    def create_table(self, schema: str, table: str, col_schemas: pd.DataFrame,
                     engine: str = 'MergeTree', partition_keys: List[str] = None, primary_keys: List[str] = None,
                     order_keys: List[str] = None,
                     table_note: str = None) -> Tuple[int, str]:

        f = ClickhouseSchema.from_dataframe(col_schemas)

        sql = f"CREATE TABLE IF NOT EXISTS {schema}.{table}" + "\n(\n" + f.to_sql() + "\n)\n" + f"ENGINE = {engine}"
        if partition_keys is not None:
            partition_keys = """PARTITION BY ({})""".format(",".join(partition_keys))
            sql += ("\n" + partition_keys)
        if primary_keys is not None:
            primary_keys = """PRIMARY KEY ({})""".format(",".join(primary_keys))
            sql += ("\n" + primary_keys)
        if order_keys is not None:
            order_keys = """ORDER BY ({})""".format(",".join(order_keys))
            sql += ("\n" + order_keys)
        if table_note is not None:
            table_note = """COMMENT '{}' """.format(table_note)
            sql += ("\n" + table_note)
        return self.client.command(cmd = sql), sql

    def create_view(self, schema: str, view: str, query: str, materialized: bool = False) -> Tuple[int, str]:
        if materialized:
            sql = f"""CREATE OR REPLACE MATERIALIZED VIEW {schema}.{view} AS"""
        else:
            sql = f"""CREATE OR REPLACE VIEW {schema}.{view} AS"""
        sql += f"\n{query}"
        return self.client.command(cmd = sql), sql

    def drop_table(self, schema: str, table: str):
        sql = "DROP TABLE {schema:Identifier}.{table:Identifier}"
        args = dict(schema=schema, table=table)
        self.client.command(cmd=sql, parameters=args)
        logging.info(f"Successfully dropped table {schema}.{table}")

    def drop_view(self, schema: str, view: str):
        sql = 'DROP VIEW {schema:Identifier}.{view:Identifier}'
        args = dict(schema=schema, view=view)
        self.client.command(cmd=sql, parameters=args)
        logging.info(f"Successfully dropped view {schema}.{view}")

    def ingest_file(self, schema: str, table: str, file_path: str, header: bool = True, use_cli: bool = False) -> int:
        fmt = self.extract_file_suffix(file_path=file_path, header=header)
        if use_cli:
            query = f"INSERT INTO {schema}.{table} FROM INFILE '{file_path}' FORMAT {fmt}"
            command = f"""{self.base_path}/usr/bin/clickhouse client --query="{query}" """
            proc = subprocess.Popen(command, shell=True)
            # proc.wait()
            logging.info(f"Submitted file ingestion task for {file_path} to {schema}.{table}")
            return command
        else:
            insert_file(
                self.client,
                table=table,
                database=schema,
                file_path=file_path,
                fmt=fmt,
            )
            logging.info(f"Successfully ingested file {file_path} to {schema}.{table}")
        return 1

    def insert_from_select(self, schema: str, table: str, query: str, qry_cols: list):
        """save the query into the table using "insert into schema.table select ... " clause

        :param schema: schema in which the table exists
        :param table: table to be inserted
        :param query: the sql query for table to be inserted
        :param qry_cols: column order matters because clickhouse use column order for mapping,
                so qry_cols should match query result column order
        :return:
        """
        sql = f"""
        INSERT INTO {schema}.{table} ({','.join(qry_cols)})
        {query}
        """
        self.client.command(cmd = sql)

    def truncate_table(self, schema: str, table: str) -> str:
        sql = f"""TRUNCATE TABLE {schema}.{table}"""
        self.client.command(cmd = sql)
        logging.info(f"Successfully truncated table {schema}.{table}")
        return sql

    def delete_data(self, schema: str, table: str, where_clause: str = None) -> str:
        sql = f"""ALTER TABLE {schema}.{table} DELETE"""
        if where_clause is not None:
            sql += f" WHERE {where_clause}"
        else:
            sql += " WHERE 1=1"
        self.client.command(cmd = sql)
        logging.info(f"Successfully deleted data")
        return sql

    def size_by_partition(self, schema: str, table: str, partition_keys: List[str]) -> pd.DataFrame:
        partitions = ",".join(partition_keys)
        sql = f"""
        SELECT
            {partitions},
            COUNT()
        FROM
            {schema}.{table}
        GROUP BY
            {partitions}
        """
        return self.client.query_df(query = sql)