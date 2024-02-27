from pathlib import Path
import os
import yaml
from botocore.exceptions import ClientError
from CommonTools.dtyper import DSchema
from CommonTools.obj_storage import ObjStorage


class BaseSchemaManager:
    def write(self, schema: str, table: str, dschema: DSchema):
        content = dschema.to_js()
        self.write_raw(schema, table, content)
    
    def write_raw(self, schema: str, table: str, content: dict):
        raise NotImplementedError("")
    
    def read(self, schema: str, table: str) -> DSchema:
        content = self.read_raw(schema, table)
        return DSchema.from_js(content)
    
    def read_raw(self, schema: str, table: str) -> dict:
        raise NotImplementedError("")
    

class LocalSchemaManager(BaseSchemaManager):
    """Table Schema Manager on local File System
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        
    def write_raw(self, schema: str, table: str, content: dict):
        # create path
        schema_folder = Path(self.root_dir) / schema
        schema_folder.mkdir(parents = True, exist_ok = True)
        # save file
        schema_file = schema_folder / f"{table}.yml"
        with open(schema_file, 'w') as obj:
            yaml.dump(content, obj, default_flow_style=False, sort_keys=False)
            
    def read_raw(self, schema: str, table: str) -> dict:
        schema_file = Path(self.root_dir) / schema / f"{table}.yml"
        try:
            with open(schema_file, 'r') as obj:
                record = yaml.safe_load(obj)
        except FileNotFoundError as e:
            raise ValueError("No record found for given schema and table")
        else:
            return record
        
        
class ObjStorageSchemaManager(BaseSchemaManager):
    """Table Schema Manager on object storage (S3, GCS, etc.)
    """
    def __init__(self, obj_st: ObjStorage, bucket: str, root_dir: str):
        obj_st.enter_bucket(bucket)
        self._obj_st = obj_st
        self.root_dir = root_dir
        
    def write_raw(self, schema: str, table: str, content: dict):
        # create content
        content_str = yaml.dump(content, default_flow_style=False, sort_keys=False)
        content_bytes = bytes(content_str, encoding = 'utf8')
        # save config
        schema_file = os.path.join(self.root_dir, schema, f"{table}.yml")
        self._obj_st.save_obj(
            body = content_bytes,
            remote_path = schema_file
        )
        
    def read_raw(self, schema: str, table: str) -> dict:
        schema_file = os.path.join(self.root_dir, schema, f"{table}.yml")
        try:
            obj = self._obj_st.read_obj(remote_path = schema_file)
            record = yaml.safe_load(obj)
        except ClientError as e:
            raise ValueError("No record found for given schema and table")
        return record
    