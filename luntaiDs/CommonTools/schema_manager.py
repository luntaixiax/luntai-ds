from pathlib import Path
from fsspec import AbstractFileSystem
import yaml
from luntaiDs.CommonTools.dtyper import DSchema


class BaseSchemaManager:
    """base interface for managing table schema metadata
    
    each schema have multiple tables, and each table have a dictionary of schema metadata
    """
    def write(self, schema: str, table: str, dschema: DSchema):
        """save the given table schema into the given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :param DSchema dschema: the table schema object
        """
        content = dschema.to_js()
        self.write_raw(schema, table, content)
    
    def write_raw(self, schema: str, table: str, content: dict):
        """handle how to write raw (dictionary) config into given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :param dict content: the dict version of Dschema object
        """
        raise NotImplementedError("")
    
    def read(self, schema: str, table: str) -> DSchema:
        """read from the system and return the assembled DSchema object

        :param str schema: the schema name
        :param str table: the table name
        :return DSchema: the table schema object
        """
        content = self.read_raw(schema, table)
        return DSchema.from_js(content)
    
    def read_raw(self, schema: str, table: str) -> dict:
        """handle how to write read (dictionary) config from given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :return dict: the dict version of Dschema object 
        """
        raise NotImplementedError("")


class SchemaManagerFileSystem(BaseSchemaManager):
    """Table Schema Manager on fsspec compatible filesystem
    """
    def __init__(self, fs: AbstractFileSystem, root_dir: str):
        """schema manager for file system
        
        folder structure:
        root_dir
            - schema1
                - table A.yml
                - table B.yml
            - schema2
                - table C.yml
                - table D.yml

        :param AbstractFileSystem fs: the fsspec compatible filesystem
        :param str root_dir: root path, if on object storage, 
            the full path including buckets
        """
        self._fs = fs
        self._root_dir = root_dir
        
    def write_raw(self, schema: str, table: str, content: dict):
        """handle how to write raw (dictionary) config into given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :param dict content: the dict version of Dschema object
        """
        # create folder
        schema_folder = Path(self._root_dir) / schema
        self._fs.makedirs(
            path = schema_folder.as_posix(),
            exist_ok = True
        )
        # save content to file
        schema_file = schema_folder / f"{table}.yml"
        with self._fs.open(schema_file, 'w') as obj:
            yaml.dump(
                content, 
                obj, 
                default_flow_style = True, 
                sort_keys = True
            )
            
    def read_raw(self, schema: str, table: str) -> dict:
        """handle how to write read (dictionary) config from given schema/table

        :param str schema: the schema name
        :param str table: the table name
        :return dict: the dict version of Dschema object 
        """
        schema_file = Path(self._root_dir) / schema / f"{table}.yml"
        try:
            with self._fs.open(schema_file, 'r') as obj:
                record = yaml.load(obj, Loader=yaml.Loader)
        except FileNotFoundError as e:
            raise ValueError("No record found for given schema and table")
        else:
            return record
    