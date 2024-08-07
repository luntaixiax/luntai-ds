import json
from pathlib import Path
from typing import List
from fsspec import AbstractFileSystem

class _BaseModelRegistry:
    """Base class for Simple Model Registry
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

    def load_config(self) -> dict:
        """load configuration
        
        :return dict: configuration in dictionary format
        """
        raise NotImplementedError("")

    def save_config(self, config:dict):
        """save configuration dictionary

        :param dict config: configuration in dictionary format
        """
        raise NotImplementedError("")

    def get_prod_model_id(self) -> str:
        """get deployed (prod) model id

        :return str: model id that is deployed
        """
        return self.load_config().get("prod")

    def get_model_list(self) -> List[str]:
        """get a list of model ids

        :return List[str]: list of model ids registered
        """
        return list(self.load_config().get("archive").keys())

    def get_model_config(self, model_id: str) -> dict:
        """get specific configuration part for a given model id

        :param str model_id: model id of the model
        :return dict: the specific configuration part
        """
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        return self.load_config().get("archive").get(model_id)
    
    def upsert_model_config(self, model_id: str, model_config: dict):
        """update or insert (if not exist) model configuration

        :param str model_id: model id of the model
        :param dict model_config: the specific configuration part
        """
        config = self.load_config()
        config['archive'][model_id] = model_config
        self.save_config(config)
        
    def delete_model_config(self, model_id: str):
        """delete model configuration

        :param str model_id: model id of the model
        """
        config = self.load_config()
        config['archive'].pop(model_id)
        if config['prod'] == model_id:
            config['prod'] = None
        self.save_config(config)

    def load_model(self, model_id: str):
        """loading the model into memory

        :param str model_id: the model id for the model
        :return _type_: model object into memory
        """
        config = self.get_model_config(model_id)
        return self.load_model_by_config(config)

    def load_prod(self):
        """load deployed model into memory

        :return _type_: deployed model object into memory
        """
        return self.load_model(self.get_prod_model_id())

    def register(self, model_id: str, *args, **kws):
        """register your model into the system

        :param str model_id: your model id
        """
        assert model_id not in self.get_model_list(), "Model Id already registered, please try another one"
        model_config = self.save_model_and_generate_config(model_id, *args, **kws)
        self.upsert_model_config(model_id, model_config)

    def remove(self, model_id: str):
        """delete model from registry

        :param str model_id: model id to be removed
        """
        assert model_id in self.get_model_list(), "Model Id not found"

        self.delete_model_files(model_id)
        self.delete_model_config(model_id)

    def deploy(self, model_id: str):
        """deploy selected model to prod, simply achieved by setting it to prod field

        :param str model_id: model id to be deployed
        """
        # check if model id exists
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        config = self.load_config()
        config["prod"] = model_id
        self.save_config(config)

    def delete_model_files(self, model_id: str):
        """delete model related files/data

        :param str model_id: model id to be deleted
        """
        raise NotImplementedError("")

    def load_model_by_config(self, config: dict):
        """load the model using configuration file

        :param dict config: configuration for the model
        """
        raise NotImplementedError("")

    def save_model_and_generate_config(self, model_id:str, *args, **kws) -> dict:
        """save model and generate configuration for this model

        :param str model_id: model id to be generated
        :return dict: the configuration generated by creating model into the system
        """
        raise NotImplementedError("")
    
    
class _BaseModelRegistryFileSystem(_BaseModelRegistry):
    """Base class for Simple Model Registry on fsspec compatible Filesystem
    Note only the configuration will be saved on the given file system, not model data
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
    def __init__(self, fs: AbstractFileSystem, config_js_path: str):
        """model registry where config is saved to given filesystem

        :param AbstractFileSystem fs: the fsspec compatible filesystem
        :param str config_js_path: path of the js config file, if on object storage, 
            the full path including buckets
        """
        self._fs = fs
        self._config_js_path = config_js_path
        self._fs.makedirs(
            path = Path(config_js_path).parent.as_posix(),
            exist_ok = True
        )

    def load_config(self) -> dict:
        """load configuration
        
        :return dict: configuration in dictionary format
        """
        if self._fs.exists(self._config_js_path):
            with self._fs.open(self._config_js_path, 'r') as obj:
                config = json.loads(obj.read())
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
        with self._fs.open(self._config_js_path, 'w') as obj:
            json.dump(config, obj, indent = 4)