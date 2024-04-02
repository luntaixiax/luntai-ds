from __future__ import annotations
from collections import OrderedDict
import json
import shutil
import os
from typing import List, Dict
import pandas as pd
from datetime import datetime
from luntaiDs.CommonTools.accessor import loadJSON, toJSON
from luntaiDs.CommonTools.obj_storage import ObjStorage

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


class _BaseModelTimeTable:
    """you can have different model (model_id) run in different time period as prod model
    support for multiple schedules
    """
    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        """load time table

        :return Dict[str, List[TimeInterval]]: time table loaded
        """
        raise NotImplementedError("")
    
    def save_tb(self, tb: Dict[str, List[TimeInterval]]):
        """save timetable

        :param Dict[str, List[TimeInterval]] tb: time table
        """
        raise NotImplementedError("")

    @ classmethod
    def sort_and_merge(cls, timetable: Dict[str, List[TimeInterval]]) -> Dict[str, List[TimeInterval]]:
        """sort and merge timetable to avoid any overlapping conflict

        :param Dict[str, List[TimeInterval]] timetable: timetable to be sorted
        :return Dict[str, List[TimeInterval]]: sorted and merged timetable
        """
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

        :param str model_id: model id to be registered
        :param datetime start: start datetime of this model id
        :param datetime end: end datetime of this model id
        :param bool force: if True, will move endpoints of existing periods to avoid overlapping
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
        """get model id given run datetime

        :param datetime dt: run datetime
        :return str: model id in place
        """
        timetable = self.load_tb()
        for mi, tis in timetable.items():
            for ti in tis:
                if dt in ti:
                    return mi
                
    def get_schedule(self) -> pd.DataFrame:
        """get runnning timetable

        :return pd.DataFrame: timetable in pandas
        """
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
    
class ModelTimeTableLocalFS:
    """you can have different model (model_id) run in different time period as prod model
    support for multiple schedules
    """
    def __init__(self, tb_js_path: str) -> None:
        self.tb_js_path = tb_js_path

    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        """load time table

        :return Dict[str, List[TimeInterval]]: time table loaded
        """
        tb = {}
        if os.path.exists(self.tb_js_path):
            c = loadJSON(self.tb_js_path)
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
        toJSON(c, self.tb_js_path)
        
        
class ModelTimeTableObjStorage:
    """you can have different model (model_id) run in different time period as prod model
    support for multiple schedules
    """ 
    def __init__(self, objstore: ObjStorage, tb_js_path: str):
        """
        
        :param tb_js_path: maybe not used if not local file system
        """
        self.tb_js_path = tb_js_path
        self._objstore = objstore

    def load_tb(self) -> Dict[str, List[TimeInterval]]:
        """load time table

        :return Dict[str, List[TimeInterval]]: time table loaded
        """
        tb = {}
        c = json.loads(
            self._objstore.read_obj(self.tb_js_path), 
            object_pairs_hook = OrderedDict
        )
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
        
        content = bytes(json.dumps(c).encode('UTF-8'))
        self._objstore.save_obj(content, self.tb_js_path)