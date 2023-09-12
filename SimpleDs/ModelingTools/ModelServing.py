from __future__ import annotations
import shutil
import os
from typing import List, Dict
import pandas as pd
from datetime import datetime
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


class TimeInterval:
    def __init__(self, start: datetime, end: datetime):
        if start >= end:
            raise ValueError("Start must be ealier than end datetime")
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"TimeInterval({self.start} - {self.end})"
    
    def __contains__(self, dt: datetime) -> bool:
        return self.start <= dt <= self.end

    def overlap(self, ti: TimeInterval) -> bool:
        return ti.end > self.start and self.end > ti.start
    
    def overlap_type(self, ti: TimeInterval) -> int:
        """judge which type is the overlapping

        there are five types
        0 - no overlapping
        1 - ti is left of self
        2 - ti is right of self
        3 - ti includes self
        4 - self includes ti
        5 - perfect overlap

        :param TimeInterval ti: _description_
        :return int: {0,1,2,3,4}
        """
        if not self.overlap(ti):
            return 0
        elif  ti.start == self.start and self.end == ti.end:
            return 5
        elif ti.start < self.start < ti.end < self.end:
            return 1
        elif self.start < ti.start < self.end < ti.end:
            return 2
        elif ti.start <= self.start < self.end <= ti.end:
            return 3
        elif self.start <= ti.start < ti.end <= self.end:
            return 4

    def to_js(self) -> Dict[str, str]:
        return {
            'start' : self.start.isoformat(), 
            'end' : self.end.isoformat()
        }
    
    @classmethod
    def from_js(cls, l: Dict[str, str]) -> TimeInterval:
        return TimeInterval(
            start = datetime.fromisoformat(l['start']),
            end = datetime.fromisoformat(l['end'])
        )


class ModelTimeTable:
    """you can have different model (model_id) run in different time period as prod model
    support for multiple schedules
    """
    def __init__(self, tb_js_path: str) -> None:
        self.tb_js_path = tb_js_path

    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        tb = {}
        if os.path.exists(self.tb_js_path):
            c = loadJSON(self.tb_js_path)
            for model_id, intervals in c.items():
                tb[model_id] = [TimeInterval.from_js(interval) for interval in intervals]
        return tb
    
    def save_tb(self, tb: Dict[str, List[TimeInterval]]):
        c = {}
        for model_id, intervals in tb.items():
            c[model_id] = [interval.to_js() for interval in intervals]
        toJSON(c, self.tb_js_path)

    @ classmethod
    def sort_and_merge(cls, timetable: Dict[str, List[TimeInterval]]) -> Dict[str, List[TimeInterval]]:
        r = {}
        for mi, tis in timetable.items():
            length = len(tis)
            if length == 1:
                r[mi] = tis
                continue

            sort_tis = sorted(tis, key=lambda x: x.start)
            s = []; i = 0
            while i < length:
                cur = sort_tis[i]
                if i + 1 < length:
                    nex = sort_tis[i + 1]
                    if cur.end == nex.start:
                        cur.end = nex.end
                        i += 1
                s.append(cur)
                i += 1

            r[mi] = s
        return r


    def register(self, model_id: str, start: datetime, end: datetime,  force: bool = False):
        """add a new time interval into the schedule

        :param str model_id: _description_
        :param datetime start: _description_
        :param datetime end: _description_
        :param bool force: if True, will move endpoints of existing periods to avoid overlapping
        :raises ValueError: _description_
        """
        new = TimeInterval(start=start, end=end)
        timetable = self.load_tb()
        for mi, tis in timetable.items():
            del_list = []
            for i, ti in enumerate(tis.copy()):
                if new.overlap(ti):
                    if force is False:
                        raise ValueError(f"Overlap with existing model {mi} serving time {ti}, set force to True or change endpoints")
                    else:
                        ti_end = ti.end
                        ti_start = ti.start
                        ovt = new.overlap_type(ti)
                        if ovt == 1:
                            # end of ti -> start of new
                            timetable[mi][i].end = start
                        elif ovt == 2:
                            # start of ti -> end of new
                            timetable[mi][i].start = end
                        elif ovt == 3:
                            # end of ti -> start of new 
                            # + create new interval [end of new, end of ti]
                            timetable[mi][i].end = start
                            timetable[mi].append(TimeInterval(start=end, end=ti_end))
                        elif ovt == 4 or ovt == 5:
                            # remove ti
                            del_list.append(i)
                            # del timetable[mi][i]  # will cause index problem when deleting
            
            # delete task
            cleaned = [timetable[mi][i] for i in range(len(timetable[mi])) if i not in del_list]
            timetable[mi] = cleaned

        
        if model_id in timetable:
            timetable[model_id].append(new)
        else:
            timetable[model_id] = [new]
        
        timetable = self.sort_and_merge(timetable)
        self.save_tb(timetable)

    def get_model_id_by_datetime(self, dt: datetime) -> str:
        timetable = self.load_tb()
        for mi, tis in timetable.items():
            for ti in tis:
                if dt in ti:
                    return mi
                
    def get_schedule(self) -> pd.DataFrame:
        timetable = self.load_tb()
        
        r =[]
        for mi, tis in timetable.items():
            for ti in tis:
                t = ti.to_js()
                t['model_id'] = mi
                r.append(t)
        
        df = pd.DataFrame.from_records(r)
        if len(df) > 0:
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            df.sort_values(by = 'start', inplace=True, ignore_index=True)

        return df
    