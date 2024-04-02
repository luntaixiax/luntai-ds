from typing import List, Dict
from datetime import datetime
import pandas as pd
import pymongo
from pymongo.mongo_client import MongoClient
from luntaiDs.ModelingTools.Serving.model_registry import _BaseModelRegistry
from luntaiDs.ModelingTools.Serving.timetable import _BaseModelTimeTable, TimeInterval
from luntaiDs.ProviderTools.mongo.serving import MongoClient

class _BaseModelRegistryMongo(_BaseModelRegistry):
    """Base class for Simple Model Registry on MongoDB
    Note Mongo only means the configuration will be saved in MongoDB, not model data
    model versioning is achieved by tracking model_id and its configuration

    Different from its super class, each model config will be a seperate entry in one collection (table)
    and prod key is added to each entry
    each model configuration format:
    {
        "model_id" : "MODEL_ID_A",
        "is_prod" : true,
        "config" : {CONFIG_A}
    }
    {
        "model_id" : "MODEL_ID_B",
        "is_prod" : false,
        "config" : {CONFIG_B}
    }
    """
    def __init__(self, mongo_client: MongoClient, db: str, collection: str):
        """
        
        :param mongo_client: the mongo client to connect with
        :param db: the database name
        :param collection: the collection (table) name for record
        """
        self._mongo = mongo_client
        self._collection = self._mongo[db][collection]
        self.db = db
        self.collection = collection
        
    def load_config(self) -> dict:
        """load configuration
        
        note that it will combine all config into 1 dictionary and convert to base format
        configuration format:
        {
            "prod" : "MODEL_ID",
            "archive" : {
                "MODEL_ID_A" : {CONFIG_A},
                "MODEL_ID_B" : {CONFIG_B},
            }
        }
        
        :return dict: configuration in dictionary format
        """
        prod_model_id = None
        archives =  {}
        for r in self._collection.find(
            {}, # find all
            {'_id': 0}, # drop id column
            sort = [( '_id', pymongo.DESCENDING )] # sort by latest record
            
        ):
            model_id = r['model_id']
            if r['is_prod']:
                prod_model_id = model_id
            archives[model_id]= r['config']

        return {
            "prod" : prod_model_id,
            "archive" : archives
        }
        
    def save_config(self, config:dict):
        """save configuration dictionary, overwrite mode
        
        note that the config here is the base format
        configuration format:
        {
            "prod" : "MODEL_ID",
            "archive" : {
                "MODEL_ID_A" : {CONFIG_A},
                "MODEL_ID_B" : {CONFIG_B},
            }
        }

        :param dict config: configuration in base dictionary format
        """
        prod_model_id = config.get('prod')
        model_ids = []
        for model_id, model_config in config.get('archive', {}).items():
            model_ids.append(model_id)
            is_prod = False
            if model_id == prod_model_id:
                is_prod = True
                
            # construct content
            content = {
                "model_id" : model_id,
                "is_prod" : is_prod,
                "config" : model_config
            }
            self._collection.replace_one(
                filter = {
                    'model_id': model_id
                }, # find matching record, if any
                replacement = content,
                upsert = True # update if found, insert if not found
            )
        
        # delete model_id that are not in config
        self._collection.delete_many(
            { 'model_id' :  { '$nin': model_ids }}
        )
        

    def get_prod_model_id(self) -> str:
        """get deployed (prod) model id

        :return str: model id that is deployed
        """
        record = (
            self._collection
            .find_one(
                {'is_prod': True}, # matching condition
                {'_id': 0, 'model_id' : 1}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        if record is None:
            return None
        return record.get('model_id')
    
    def get_model_list(self) -> List[str]:
        """get a list of model ids

        :return List[str]: list of model ids registered
        """
        return self._collection.distinct('model_id')
    
    def get_model_config(self, model_id: str) -> dict:
        """get specific configuration part for a given model id

        :param str model_id: model id of the model
        :return dict: the specific configuration part
        """
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        record = (
            self._collection
            .find_one(
                {'model_id': model_id}, # matching condition
                {'_id': 0, 'config' : 1}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        return record.get('config')
        
    def upsert_model_config(self, model_id: str, model_config: dict):
        """update or insert (if not exist) model configuration

        :param str model_id: model id of the model
        :param dict model_config: the specific configuration part
        """
        is_prod_field = (
            self._collection
            .find_one(
                {'model_id': model_id}, # matching condition
                {'_id': 0, 'is_prod' : 1}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        if is_prod_field is None:
            is_prod = False
        else:
            is_prod = is_prod_field.get('is_prod', False)
            
        # construct content
        content = {
            "model_id" : model_id,
            "is_prod" : is_prod,
            "config" : model_config
        }
        self._collection.replace_one(
            filter = {
                'model_id': model_id
            }, # find matching record, if any
            replacement = content,
            upsert = True # update if found, insert if not found
        )
        
    def delete_model_config(self, model_id: str):
        """delete model configuration

        :param str model_id: model id of the model
        """
        self._collection.delete_one({
            'model_id' : model_id
        })

    def deploy(self, model_id: str):
        """deploy selected model to prod, simply achieved by setting it to prod field

        :param str model_id: model id to be deployed
        """
        # check if model id exists
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        # set all existing prod to False
        self._collection.update_many(
            {
                'is_prod': True
            }, # filter: find matching record, if any
            { 
                "$set": { 'is_prod': False } 
            } # update content 
        )
        # set selected model id to have True prod state
        self._collection.update_one(
            {
                'model_id': model_id
            }, # filter: find matching record, if any
            { 
                "$set": { 'is_prod': True } 
            } # update content 
        )
        
        

class ModelTimeTableMongo(_BaseModelTimeTable):
    def __init__(self, mongo_client: MongoClient, db: str, collection: str):
        self._mongo = mongo_client
        self._collection = self._mongo[db][collection]
        
    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        """load time table

        :return Dict[str, List[TimeInterval]]: time table loaded
        """
        result = {}
        for record in self._collection.find({}):
            model_id = record.get('model_id')
            ti = TimeInterval(
                start = record.get('start'), 
                end = record.get('end')
            )
            if model_id in result:
                result[model_id].append(ti)
            else:
                result[model_id] = [ti]
        return result
                
    
    def save_tb(self, tb: Dict[str, List[TimeInterval]]):
        """save timetable

        :param Dict[str, List[TimeInterval]] tb: time table
        """
        # need to overwrite everything, so need to delete all first
        self._collection.delete_many({})
        
        for model_id, intervals in tb.items():
            for interval in intervals:
                self._collection.insert_one(
                    {
                        'model_id' : model_id,
                        'start' : interval.start,
                        'end' : interval.end
                    }
                )
                
    def get_model_id_by_datetime(self, dt: datetime) -> str:
        """get model id given run datetime

        :param datetime dt: run datetime
        :return str: model id in place
        """
        record = (
            self._collection
            .find_one(
                {'start': {'$lte' : dt}, 'end' : {'$gte' : dt}}, # matching condition
                {'_id': 0}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        if record is None:
            return None
        return record.get('model_id')
    
    def get_schedule(self) -> pd.DataFrame:
        """get runnning timetable

        :return pd.DataFrame: timetable in pandas
        """
        return pd.DataFrame(
            self._collection
            .find({}, { "_id": 0})
            .sort('start')
        )