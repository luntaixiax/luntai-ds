from __future__ import annotations
from datetime import date
import logging
import time
from typing import List, Literal
from collections import OrderedDict
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerBase
from luntaiDs.CommonTools.SnapStructure.tools import get_future_period_ends, get_past_period_ends
from luntaiDs.CommonTools.utils import str2dt, offset_date

""" Chain Executor with dependency check """

class _CurrentStream:
    def __init__(self, upstream, offset: int = 0, 
                 freq: Literal['M', 'D', 'W'] = 'M'):
        self.upstream = upstream
        self.offset = offset
        self.freq = freq

    def get_exc_dts(self, snap_dt: date):
        if self.offset != 0:
            snap_dt = offset_date(snap_dt, offset = self.offset, freq = self.freq)
        return [snap_dt]


class _FutureStream(_CurrentStream):
    def __init__(self, upstream, future: int, offset: int = 0, 
                 freq: Literal['M', 'D', 'W'] = 'M'):
        super().__init__(upstream)
        self.future = future
        self.freq = freq
        self.offset = offset

    def get_exc_dts(self, snap_dt: date):
        if self.offset != 0:
            snap_dt = offset_date(snap_dt, offset = self.offset, freq = self.freq)
        return get_future_period_ends(snap_dt, forward=self.future, freq = self.freq)


class _PastStream(_CurrentStream):
    def __init__(self, upstream, history: int, offset: int = 0, 
                 freq: Literal['M', 'D', 'W'] = 'M'):
        super().__init__(upstream)
        self.history = history
        self.freq = freq
        self.offset = offset

    def get_exc_dts(self, snap_dt: date):
        if self.offset != 0:
            snap_dt = offset_date(snap_dt, offset = self.offset, freq = self.freq)
        return get_past_period_ends(snap_dt, history=self.history, freq = self.freq)[::-1]
    

class ExecPlan:
    def __init__(self, obj, snap_dt: date):
        self.obj = obj
        self.snap_dt = snap_dt
        self.upstreams = []

    def __str__(self):
        return f"{self.obj}@{self.snap_dt} | upstreams = {len(self.upstreams)}"

    def attach_upstream(self, plan: ExecPlan):
        self.upstreams.append(plan)

    def exec_plan(self) -> dict:
        up_infos = []
        for up in self.upstreams:
            up_infos.append(up.exec_plan())

        return {
            'current': self,
            'upstream' : up_infos
        }
    
    def get_serial_steps(self) -> list:
        """get a serias of steps to execute

        :return list: _description_
        """
        r = []
        for up in self.upstreams:
            r.extend(up.get_serial_steps())
        r.append(self)
        return r
    
    def reduce_summary(self) -> OrderedDict:
        """combine similar steps (same obj and snap date) from serial plan and do summarize

        :return dict: _description_
        """
        summary = OrderedDict()
        steps = self.get_serial_steps()
        for step in steps:
            if step.obj in summary.keys():
                if step.snap_dt not in summary[step.obj]:
                    summary[step.obj].append(step.snap_dt)
            else:
                summary[step.obj] = [step.snap_dt]
        return summary



class SnapTableStreamGenerator:
    dm: SnapshotDataManagerBase = None
    upstreams: List[_CurrentStream] = []
    
    @classmethod
    def init(cls, snap_dt: date):
        """do some initialization if not exist or ready

        :param date snap_dt: _description_
        """
        pass

    @classmethod
    def execute(cls, snap_dt: date):
        """key logic for generating data at given snap date

        :param date snap_dt: _description_
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError("")
    
    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        """determine if the run is successful or not

        :param date snap_dt: _description_
        :return bool: if True will run next job/upstream, otherwise will delete and rerun
        """
        return cls.dm.is_valid(snap_dt = snap_dt)

    @classmethod
    def _execute(cls, snap_dt: date):
        """execute the excute function defined

        :param date snap_dt: _description_
        """
        logging.info(f"Running {cls.dm}@{snap_dt}")
        snap_dt = str2dt(snap_dt)
        # single run, without checking the upstream tables

        start = time.time()
        cls.execute(snap_dt = snap_dt)
        end = time.time()
        seconds = end - start
        logging.info(f"Saved to {cls.dm}@{snap_dt}, used time {seconds // 60 : .0f} minutes {seconds % 60 : .0f} seconds")

    
    @classmethod
    def _prerun(cls, snap_dt: date):
        """preparation before actual run
        

        :param date snap_dt: _description_
        :return _type_: _description_
        """
        if cls.dm.exist():
            if snap_dt in cls.dm.get_existing_snap_dts():
                if cls.if_success(snap_dt):
                    logging.info(f"Already extracted for {cls.dm}@{snap_dt}, will bypass")
                    return
                else:
                    logging.warning(f"Existing records not valid for {cls.dm}@{snap_dt}, will rerun")
                    cls.dm.delete(snap_dt)
        else:
            logging.info(f"Table-Schema does not exist, will initialize")
            cls.init(snap_dt)
    
    @classmethod
    def run(cls, snap_dt: date):
        """run current and upstreams

        :param date snap_dt: _description_
        """
        snap_dt = str2dt(snap_dt)
        # prepare
        cls._prerun(snap_dt)

        # check upstream table recursively and make up any missing files
        for up in cls.upstreams:
            logging.info(f"Upstream found: {up.upstream.dm}")
            for dt in up.get_exc_dts(snap_dt):
                logging.info(f"Run upstream {up.upstream.dm} @ {dt}")
                up.upstream.run(dt)

        # execute
        cls._execute(snap_dt)
        
    @classmethod
    def run_detached(cls, snap_dt: date):
        """run current timestamp only, ignore upstreams

        :param date snap_dt: _description_
        :return _type_: _description_
        """
        snap_dt = str2dt(snap_dt)
        # prepare
        cls._prerun(snap_dt)
            
        # execute
        cls._execute(snap_dt)
            

    @classmethod
    def get_exec_plan(cls, snap_dt: date) -> ExecPlan:
        """get the execution plan object for further analysis

        :param date snap_dt: _description_
        :return ExecPlan: _description_
        """
        snap_dt = str2dt(snap_dt)

        plan = ExecPlan(
            obj = cls,
            snap_dt = snap_dt
        )

        for up in cls.upstreams:
            for dt in up.get_exc_dts(snap_dt):
                plan.attach_upstream(up.upstream.get_exec_plan(dt))

        return plan