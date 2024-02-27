import pymongo
from collections import OrderedDict
from pymongo.mongo_client import MongoClient
from CommonTools.schema_manager import BaseSchemaManager
from CommonTools.dtyper import DSchema


class MongoSchemaManager(BaseSchemaManager):
    
    def __init__(self, mongo_client: MongoClient, database: str, collection: str):
        self._mongo_client = mongo_client
        self.database = database
        self.collection = collection
    
    def write_raw(self, schema: str, table: str, content: dict):
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