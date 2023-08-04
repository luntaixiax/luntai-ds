import shutil
import os
from typing import List
import pandas as pd
from CommonTools.accessor import loadJSON, toJSON

class ModelRegistry:
    def __init__(self, config_js_path: str):
        """

        :param config_js_path:

        js format:
        {
            "prod" : "MODEL_ID",
            "archive" : {
                "MODEL_ID_A" : {CONFIG_A},
                "MODEL_ID_B" : {CONFIG_B},
            }
        }
        """
        self.config_js_path = config_js_path

    def load_config(self) -> dict:
        if os.path.exists(self.config_js_path):
            config = loadJSON(self.config_js_path)
        else:
            config = {
                "prod" : None,
                "archive": {}
            }
        return config

    def save_config(self, config:dict):
        toJSON(config, self.config_js_path)

    def get_prod_model_id(self) -> str:
        return self.load_config().get("prod")

    def get_model_list(self) -> List[str]:
        return list(self.load_config().get("archive").keys())

    def get_model_config(self, model_id: str) -> dict:
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        return self.load_config().get("archive").get(model_id)

    def load_model(self, model_id: str):
        config = self.get_model_config(model_id)
        return self.load_model_by_config(config)

    def load_prod(self):
        return self.load_model(self.get_prod_model_id())

    def register(self, model_id: str, *args, **kws):
        assert model_id not in self.get_model_list(), "Model Id already registered, please try another one"
        model_config = self.save_model_and_generate_config(model_id, *args, **kws)
        config = self.load_config()
        config['archive'][model_id] = model_config
        self.save_config(config)

    def remove(self, model_id: str):
        assert model_id in self.get_model_list(), "Model Id not found"

        config = self.load_config()
        self.delete_model_files(model_id)
        config['archive'].pop(model_id)
        if config['prod'] == model_id:
            config['prod'] = None
        self.save_config(config)

    def deploy(self, model_id: str):
        # check if model id exists
        assert model_id in self.get_model_list(), "Model Id not found in registry, please register one first"
        config = self.load_config()
        config["prod"] = model_id
        self.save_config(config)

    def delete_model_files(self, model_id: str):
        raise NotImplementedError("")

    def load_model_by_config(self, config: dict):
        raise NotImplementedError("")

    def save_model_and_generate_config(self, model_id:str, *args, **kws) -> dict:
        raise NotImplementedError("")

class ModelDataRegistry:
    def __init__(self, data_root: str):
        """
        data_root
        - data_id
            - train.parquet
            - test.parquet
        """
        self.data_root = data_root
        os.makedirs(data_root, exist_ok=True)

    def get_existing_ids(self):
        return os.listdir(self.data_root)

    def get_train_path(self, data_id: str):
        return os.path.join(self.data_root, data_id, f"train_{data_id}.parquet")

    def get_test_path(self, data_id: str):
        return os.path.join(self.data_root, data_id, f"test_{data_id}.parquet")

    def register(self, data_id: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame, replace: bool = False):
        existing_ids = self.get_existing_ids()
        if not replace and data_id in existing_ids:
            raise ValueError(f"data id {data_id} already exist, pls use another id")
        os.makedirs(os.path.join(self.data_root, data_id), exist_ok=False)
        train_ds.to_parquet(self.get_train_path(data_id))
        test_ds.to_parquet(self.get_test_path(data_id))

    def fetch(self, data_id: str, target_col: str = None):
        assert data_id in self.get_existing_ids(), f"data id {data_id} does not exist"
        train_ds = pd.read_parquet(self.get_train_path(data_id))
        test_ds = pd.read_parquet(self.get_test_path(data_id))
        if target_col:
            X_train = train_ds.drop(columns=[target_col]).reset_index(drop=True)
            X_test = test_ds.drop(columns=[target_col]).reset_index(drop=True)
            y_train = train_ds[target_col].reset_index(drop=True)
            y_test = test_ds[target_col].reset_index(drop=True)
            return X_train, y_train, X_test, y_test
        else:
            return train_ds, test_ds

    def remove(self, data_id: str):
        dpath = os.path.join(self.data_root, data_id)
        shutil.rmtree(dpath)