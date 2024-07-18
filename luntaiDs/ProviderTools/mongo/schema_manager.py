import pymongo
from pymongo.mongo_client import MongoClient
from luntaiDs.CommonTools.schema_manager import BaseSchemaManager


class MongoSchemaManager(BaseSchemaManager):
    
    def __init__(self, mongo_client: MongoClient, database: str, collection: str):
        """schema manager for mongo db
        
        the mongodb implementation works like this, under given database/collection
        mongodb
            database
                collection
                    entry1: schema A, table 1, config
                    entry2: schema A, table 2, config
                    entry3: schema B, table 1, config
                    ...

        :param MongoClient mongo_client: mongo db python connector object
        :param str database: the mongo db database name to save the table config
        :param str collection: the mongo db table name to save the table config
        """
        self._mongo_client = mongo_client
        self.database = database
        self.collection = collection
    
    def write_raw(self, schema: str, table: str, content: dict):
        """handle how to write raw (dictionary) config into given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :param dict content: the dict version of Dschema object
        """
        # add more keys
        content = {
            'schema': schema, 
            'table': table,
            'columns' : content,
        }
        db = self._mongo_client[self.database]
        collection = db[self.collection]
        collection.replace_one(
            filter = {
                'schema': schema,
                'table': table
            }, # find matching record, if any
            replacement = content,
            upsert = True # update if found, insert if not found
        )
    
    def read_raw(self, schema: str, table: str) -> dict:
        """handle how to write read (dictionary) config from given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :return dict: the dict version of Dschema object 
        """
        db = self._mongo_client[self.database]
        collection = db[self.collection]
        record = (
            collection
            .find_one(
                {'schema': schema,'table': table,}, # matching condition
                {'_id': 0,}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        if record is None:
            raise ValueError("No record found for given schema and table")
        return record.get('columns') # only need the columns field