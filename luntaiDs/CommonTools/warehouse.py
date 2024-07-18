import ibis
import pandas as pd
from luntaiDs.CommonTools.dbapi import baseDbInf
from luntaiDs.CommonTools.dtyper import DSchema

class BaseWarehouseHandler:
    @classmethod
    def connect(cls, db_conf: baseDbInf, **settings):
        cls._db_conf = db_conf
        
    def is_exist(self, schema: str, table: str) -> bool:
        """whether the schema and table exist, or ready to do operations
        for DB based, usually it detects whether the table shema structure is created
        
        :param str schema: schema/database
        :param str table: table name
        :return bool: whether the table and schema exists and ready
        """
        raise NotImplementedError("")
    
    def get_table(self, schema: str, table: str) -> ibis.expr.types.Table:
        """get the ibis table
        
        :param str schema: schema/database
        :param str table: table name
        """
        raise NotImplementedError("")
    
    def create_schema(self, schema: str):
        """create database/schema
        
        :param str schema: schema/database
        """
        raise NotImplementedError("")
        
    def create_table(self, schema: str, table: str, col_schemas: DSchema, **settings):
        """initialize/create table in the underlying data warehouse system

        :param str schema: schema/database
        :param str table: table name
        :param DSchema col_schemas: data column schema
        :param List[str] primary_keys: primary keys, defaults to None
        """
        raise NotImplementedError("")
            
    def get_dtypes(self, schema: str, table: str) -> ibis.Schema:
        """Check dtypes of the given schema/dataset

        :return:
        """
        raise NotImplementedError("")
        
    def query(self, sql: str, schema: str = None) -> ibis.expr.types.Table:
        """query using SQL

        :return: ibis dataframe
        """
        raise NotImplementedError("")
    
    def execute(self, sql: str, **kws):
        """execute SQL for operations like insert/update/delete

        :return: ibis dataframe
        """
        raise NotImplementedError("")
        
    def save_pandas(self, df: pd.DataFrame, schema: str, table: str, **kws):
        """The pure logic to save pandas dataframe to the system, without handling existing record problem

        :param pd.DataFrame df: pandas table
        :param str schema: schema/database
        :param str table: table name
        """
        raise NotImplementedError("")
    
    def save_ibis(self, df: ibis.expr.types.Table, schema: str, table: str, **kws):
        """The pure logic to save ibis dataframe to the system, without handling existing record problem

        :param ibis.expr.types.Table df: ibis table
        :param str schema: schema/database
        :param str table: table name
        """
        raise NotImplementedError("")
        
    def delete_table(self, schema: str, table: str):
        """drop whole table
        
        :param str schema: schema/database
        :param str table: table name
        """
        raise NotImplementedError("")
    
    def truncate_table(self, schema: str, table: str):
        """Truncate (remove) data from table
        
        :param str schema: schema/database
        :param str table: table name
        """
        raise NotImplementedError("")