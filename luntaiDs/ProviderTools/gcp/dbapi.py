import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import ibis
import pandas as pd
import pyarrow as pa
from google.oauth2 import service_account
from luntaiDs.CommonTools.dbapi import baseDbInf
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.warehouse import BaseWarehouseHandler

class BigQuery(baseDbInf):
    def __init__(self, project_id: str, driver="bigquery", credentials_path: str = None):
        """
        bq = BigQuery(project_id='your-project-id')
        bq.bindServer(db = 'your-dataset-id')
        bq.launch()
        """
        super().__init__(driver)
        self.project_id = project_id
        self.credentials_path = credentials_path # json file
        self.bindServer(db = None)

    def argTempStr(self):
        return "%s"

    def getJdbcUrl(self) -> str:
        return f"jdbc:bigquery://{self.ip}:{self.port};ProjectId={self.project_id}"

    def getDriverClass(self) -> str:
        # https://mvnrepository.com/artifact/com.google.cloud/google-cloud-bigquery/2.37.1
        return "com.simba.googlebigquery.jdbc42.Driver"

    def getConnStr(self) -> str:
        # only used for sqlalchemy driver type
        return f"{self.driver}://{self.project_id}/{self.db}"
    
    def bindServer(self, ip: str = 'https://www.googleapis.com/bigquery/v2', port: int = 443, db: str = None):
        """Connect to a database server

        :param ip: ip of the server
        :param port: port number of the server
        :param db: which database to connect
        :return:
        """
        self.ip = ip
        self.port = port
        self.db = db

    def launch(self):
        """Launch the databse connector, create the sqlalchemy engine and create a session

        :return:
        """
        connStr = self.getConnStr()
        self.engine = create_engine(connStr, credentials_path=self.credentials_path)
        self.DBSession = sessionmaker(bind = self.engine)
        logging.info("Engine started, ready to go!")
        

class WarehouseHandlerBQSQL(BaseWarehouseHandler):
    @classmethod
    def connect(cls, db_conf: BigQuery, **settings):
        cls._db_conf = db_conf
        if db_conf.credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                db_conf.credentials_path
            )
        else:
            creds = None
        cls._ops = ibis.bigquery.connect(
            project_id=db_conf.project_id,
            dataset_id=db_conf.db,
            credentials = creds if creds else None,
            **settings
        )
        
    def is_exist(self, schema: str, table: str) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created
        
        :param str schema: schema/database
        :param str table: table name
        :return bool: whether the table and schema exists and ready
        """
        if schema not in self._ops.list_schemas(like = schema):
            return False
        return table in self._ops.list_tables(like = table, schema = schema)
    
    def get_table(self, schema: str, table: str) -> ibis.expr.types.Table:
        """get the ibis table
        
        :param str schema: schema/database
        :param str table: table name
        """
        return self._ops.table(name = table, schema = schema)
    
    def create_schema(self, schema: str):
        """create database/schema
        
        :param str schema: schema/database
        """
        self._ops.create_schema(
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
        # create table
        primary_keys = col_schemas.primary_keys
        partition_keys = col_schemas.partition_keys
        cluster_keys = col_schemas.cluster_keys
        logging.info(f"Creating table {schema}.{table} using schema:\n{col_schemas.ibis_schema}")
        logging.info(f"primary keys = {primary_keys}, partition_keys = {partition_keys}")
        # create table
        self._ops.create_table(
            name = table,
            schema = col_schemas.ibis_schema,
            database = schema,
            overwrite = False,
            partition_by = partition_keys,
            cluster_by = cluster_keys,
            options = settings
        )
        # add column descriptions
        descrs = col_schemas.descrs
        for col, descr in descrs.items():
            try:
                sql = f"""
                ALTER TABLE {schema}.{table} 
                ALTER COLUMN {col} 
                SET OPTIONS (
                    description="{descr}"
                )"""
                self._ops.con.command(sql)
            except:
                pass
            
    def get_dtypes(self, schema: str, table: str) -> ibis.Schema:
        """Check dtypes of the given schema/dataset

        :return:
        """
        return self._ops.get_schema(
            name = table,
            schema = schema
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
        self._ops.raw_sql(sql, **kws)
        
    def save_pandas(self, df: pd.DataFrame, schema: str, table: str, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: _description_
        :param str schema: schema/database
        :param str table: table name
        """
        self._ops.client.load_table_from_dataframe(
            dataframe = df,
            destination = f"{schema}.{table}",
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
            schema = schema,
            force = True
        )
    
    def truncate_table(self, schema: str, table: str):
        """Truncate (remove) data from table
        
        :param str schema: schema/database
        :param str table: table name
        """
        self._ops.truncate_table(
            name = table,
            schema = schema
        )