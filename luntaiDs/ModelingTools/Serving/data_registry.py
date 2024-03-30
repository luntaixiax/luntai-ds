import os
import shutil
from typing import List, Dict, Union
import ibis
import pandas as pd

class _BaseModelDataRegistry:

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        raise NotImplementedError("")

    def register(self, data_id: str, train_ds: Union[ibis.expr.types.Table | pd.DataFrame], 
                 test_ds: Union[ibis.expr.types.Table | pd.DataFrame], replace: bool = False):
        """register table in ibis format

        :param str data_id: data id
        :param Union[ibis.expr.types.Table | pd.DataFrame] train_ds: training dataset
        :param Union[ibis.expr.types.Table | pd.DataFrame] test_ds: testing dataset
        :param bool replace: whether to replace existing dataset, defaults to False
        """
        raise NotImplementedError("")

    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        raise NotImplementedError("")

    def remove(self, data_id: str):
        """remove dataset from registry

        :param str data_id: data id to be removed
        """
        raise NotImplementedError("")


class ModelDataRegistryLocalFS(_BaseModelDataRegistry):
    def __init__(self, data_root: str):
        """
        data_root
        - data_id
            - train.parquet
            - test.parquet
        """
        self.data_root = data_root
        os.makedirs(data_root, exist_ok=True)

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        return os.listdir(self.data_root)

    def get_train_path(self, data_id: str) -> str:
        return os.path.join(self.data_root, data_id, f"train_{data_id}.parquet")

    def get_test_path(self, data_id: str) -> str:
        return os.path.join(self.data_root, data_id, f"test_{data_id}.parquet")

    def register(self, data_id: str, train_ds: ibis.expr.types.Table, test_ds: ibis.expr.types.Table, replace: bool = False):
        """register table in ibis format

        :param str data_id: data id
        :param ibis.expr.types.Table train_ds: training dataset
        :param ibis.expr.types.Table test_ds: testing dataset
        :param bool replace: whether to replace existing dataset, defaults to False
        """
        existing_ids = self.get_existing_ids()
        if not replace and data_id in existing_ids:
            raise ValueError(f"data id {data_id} already exist, pls use another id")
        os.makedirs(os.path.join(self.data_root, data_id), exist_ok=False)
        train_ds.to_parquet(self.get_train_path(data_id))
        test_ds.to_parquet(self.get_test_path(data_id))

    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        assert data_id in self.get_existing_ids(), f"data id {data_id} does not exist"
        train_ds = ibis.read_parquet(self.get_train_path(data_id))
        test_ds = ibis.read_parquet(self.get_test_path(data_id))
        if target_col:
            X_train = train_ds.drop(columns=[target_col])
            X_test = test_ds.drop(columns=[target_col])
            y_train = train_ds[target_col]
            y_test = test_ds[target_col]
            return X_train, y_train, X_test, y_test
        else:
            return train_ds, test_ds

    def remove(self, data_id: str):
        """remove dataset from registry

        :param str data_id: data id to be removed
        """
        dpath = os.path.join(self.data_root, data_id)
        shutil.rmtree(dpath)
