import os
import json
from typing import List, Dict
from collections import OrderedDict
from luntaiDs.ModelingTools.Serving.model_registry import _BaseModelRegistry
from luntaiDs.ModelingTools.Serving.timetable import TimeInterval
from luntaiDs.ProviderTools.ssh.sftp import SFTP

class _BaseModelRegistrySFTP(_BaseModelRegistry):
    """Base class for Simple Model Registry on SFTP Filesystem
    Note SFTP only means the configuration will be saved in SFTP system, not model data
    model versioning is achieved by tracking model_id and its configuration

    configuration format:
    {
        "prod" : "MODEL_ID",
        "archive" : {
            "MODEL_ID_A" : {CONFIG_A},
            "MODEL_ID_B" : {CONFIG_B},
        }
    }
    """
    def __init__(self, sftp: SFTP, config_js_path: str):
        """
        
        :param config_js_path: maybe not used if not local file system
        """
        self.config_js_path = config_js_path
        self._sftp = sftp
        self._sftp.mkdir(os.path.dirname(config_js_path), ignore_existing = True)

    def load_config(self) -> dict:
        """load configuration
        
        :return dict: configuration in dictionary format
        """
        if self._sftp.exist(self.config_js_path):
            with self._sftp.getFileHandler(self.config_js_path) as obj:
                config = json.load(obj, object_pairs_hook = OrderedDict)
        else:
            config = {
                "prod" : None,
                "archive": {}
            }
        return config

    def save_config(self, config:dict):
        """save configuration dictionary

        :param dict config: configuration in dictionary format
        """
        with self._sftp.getFileHandler(self.config_js_path, 'w') as obj:
            json.dump(config, obj, indent = 4)
            
            
class ModelTimeTableSFTP:
    """you can have different model (model_id) run in different time period as prod model
    support for multiple schedules
    """
    def __init__(self, sftp: SFTP, tb_js_path: str):
        """
        
        :param tb_js_path: maybe not used if not local file system
        """
        self.tb_js_path = tb_js_path
        self._sftp = sftp
        self._sftp.mkdir(os.path.dirname(tb_js_path), ignore_existing = True)

    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        """load time table

        :return Dict[str, List[TimeInterval]]: time table loaded
        """
        tb = {}
        if self._sftp.exist(self.tb_js_path):
            with self._sftp.getFileHandler(self.tb_js_path) as obj:
                c = json.load(obj, object_pairs_hook = OrderedDict)
            
            for model_id, intervals in c.items():
                tb[model_id] = [TimeInterval.from_js(interval) for interval in intervals]
        return tb
    
    def save_tb(self, tb: Dict[str, List[TimeInterval]]):
        """save timetable

        :param Dict[str, List[TimeInterval]] tb: time table
        """
        c = {}
        for model_id, intervals in tb.items():
            c[model_id] = [interval.to_js() for interval in intervals]

        with self._sftp.getFileHandler(self.tb_js_path, 'w') as obj:
            json.dump(c, obj, indent = 4)