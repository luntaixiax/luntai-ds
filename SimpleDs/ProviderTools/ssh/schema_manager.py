import os
import yaml
from CommonTools.schema_manager import BaseSchemaManager
from ProviderTools.ssh.sftp import SFTP


class SFTPSchemaManager(BaseSchemaManager):
    """Table Schema Manager on SFTP server
    """
    def __init__(self, sftp: SFTP, root_dir: str):
        self._sftp = sftp
        self.root_dir = root_dir
        self._sftp.mkdir(
            path = self.root_dir,
            ignore_existing = True
        )
        
    def write_raw(self, schema: str, table: str, content: dict):
        # create path
        schema_folder = os.path.join(self.root_dir, schema)
        self._sftp.mkdir(
            path = schema_folder,
            ignore_existing = True
        )
        # save file
        schema_file = os.path.join(schema_folder, f"{table}.yml")
        with self._sftp.getFileHandler(schema_file, 'w') as obj:
            yaml.dump(content, obj, default_flow_style=False, sort_keys=False)
        
    def read_raw(self, schema: str, table: str) -> dict:
        schema_file = os.path.join(self.root_dir, schema, f"{table}.yml")
        try:
            with self._sftp.getFileHandler(schema_file, 'r') as obj:
                record = yaml.safe_load(obj)
        except FileNotFoundError as e:
            raise ValueError("No record found for given schema and table")
        else:
            return record