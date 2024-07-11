from datetime import date, timedelta
import glob
import os
from typing import List
import re
import pandas as pd
from luntaiDs.CommonTools.utils import get_month_end_date

def get_forward_month_end_date(dt: date) -> date:
    next_mth_start = get_month_end_date(dt.year, dt.month) + timedelta(days=1)
    return get_month_end_date(next_mth_start.year, next_mth_start.month)

def get_last_month_end_date(dt: date) -> date:
    return dt.replace(day=1) - timedelta(days=1)

def get_file_list_pattern(folder_path: str, pattern: str, include_dir: bool = True) -> list:
    """return a list of filepath that matches the pattern

    :param folder_path:
    :param pattern:
    :param include_dir: whether to include the dirname in the result
    :return: a list of macthed file paths (including/excluding directory path)

    >>> get_file_list_pattern("/app/CMM_BFS_CRI_BRR/Model_Dev_2022/Model_Dev_Pipeline/1_raw_data_extraction", pattern = "*clnt*", include_dir = False)
    """
    files_with_dir = glob.glob(f"{folder_path}/{pattern}")
    if include_dir:
        return files_with_dir
    else:
        return [os.path.basename(f) for f in files_with_dir]


def match_date_from_str(s:str) -> str:
    """

    :param s: string, date should be in YYYY-MM-DD format or YYYY-DD-MM format
    :return:
    """
    m = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", s)
    if m:
        return m.group()
    else:
        return

def generate_cohort_list(start_month: date, end_month: date, freq: str = 'M') -> List[date]:
    """

    :param start_month: start date of the cohort
    :param end_month: end date of the cohort
    :param freq: # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return:

    https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    """
    ts = pd.date_range(start = start_month, end = end_month, freq = freq).to_pydatetime()
    return [t.date() for t in ts]

def get_past_period_ends(snap_dt: date, history:int = 12, freq: str = 'M') -> List[date]:
    # return a list of month end date for the past 12 months prior to the snap_dt
    ts = pd.date_range(end = snap_dt, freq = freq, periods = history).to_pydatetime()
    return [t.date() for t in ts]

def get_future_period_ends(snap_dt: date, forward:int = 12, freq: str = 'M') -> List[date]:
    # return a list of month end date for the past 12 months prior to the snap_dt
    ts = pd.date_range(start = snap_dt, freq = freq, periods = forward).to_pydatetime()
    return [t.date() for t in ts]