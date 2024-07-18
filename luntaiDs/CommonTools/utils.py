import logging
import os
import shutil
from typing import Literal, Tuple, List, Union
from calendar import monthrange
import uuid
from fsspec import AbstractFileSystem
import pyarrow as pa
import pyarrow.parquet as pq
import ibis
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import time
from tqdm.auto import tqdm


def str2dt(dt_str: Union[str, date], format = "%Y-%m-%d") -> date:
    if isinstance(dt_str, str):
        return datetime.strptime(dt_str, format).date()
    if isinstance(dt_str, datetime):
        return dt_str.date()
    if isinstance(dt_str, date):
        return dt_str
    else:
        return dt_str

def dt2str(dt: Union[str, date], format = "%Y-%m-%d") -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, (date, datetime)):
        return dt.strftime(format)
    else:
        return dt
    
def get_month_end_date(snap_year:int, snap_month:int) -> date:
    month_end_date = date(
        snap_year, snap_month,
        monthrange(snap_year, snap_month)[1]
    )
    return month_end_date

def offset_date(base_date: date, offset: int, freq: Literal['M', 'D', 'W']):
    if freq == 'M':
        return base_date + relativedelta(months=offset)
    elif freq == 'D':
        return base_date + relativedelta(days=offset)
    elif freq == 'W':
        return base_date + relativedelta(weeks=offset)
    else:
        raise ValueError("freq can only be W/D/M")

def save_xls(names, list_dfs, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            print(names[n])
            df.to_excel(writer, names[n], index = False)
        #             df.to_excel(writer,'sheet%s' % n, index=False)
        writer.save()


def read_sql_from_file(file: str) -> str:
    with open(file) as obj:
        sql = obj.read()
    return sql

def complie_sql(sql : str, *args) -> str:
    # fill args into sql templates with ?
    arg_num_sql = sql.count("?")
    if len(args) != arg_num_sql:
        raise ValueError(f"SQL temp contains {arg_num_sql} placeholders(?) while only {len(args)} parsed")
    return sql.replace("?", "'{}'").format(*args)

def render_sql_from_file(file: str, **fill_params) -> str:
    with open(file) as obj:
        sql = obj.read()
    if fill_params:
        sql = sql.format(**fill_params)
    return sql

def df_strip(data):
    """
    Remove leading and trailing blank for each column

    Parameters:
    data : DataFrame

    Returns : DataFrame
    """
    for var in data.columns:
        if data[var][data[var].notna()].shape[0] == 0:
            pass
        else:
            if type(data[var][data[var].notna()].iloc[0]) is str:
                data[var] = data[var].str.strip()
    return data



def missing_val_chk(data, var):
    if data[var].isnull().any(axis=None):
        print('There are ' + str(data[var].isnull().sum()) + ' missing values for column ' + var + '.')
        string1 = ','.join(['There are ', str(data[var].isnull().sum()), ' missing values for column ', var, '.'])
        #string1=" "
    else:
        print('Column ' + var + ' has no missing value.')
        string1 = 'Column ' + var + ' has no missing value.'
        #string1=" i"
    return string1


def download_jdbc_jar(jar_file_url, jdbc_driver_folder, jar_file_name: str = None):
    if not os.path.exists(jdbc_driver_folder):
        os.makedirs(jdbc_driver_folder)

    if not jar_file_name:
        jar_file_name = jar_file_url.split('/')[-1]

    jar_path = os.path.join(jdbc_driver_folder, jar_file_name)
    with requests.get(
            jar_file_url,
            # proxies = {
            #     'http': SETTINGS.HTTP_PROXY,
            #     'https': SETTINGS.HTTPS_PROXY
            # }
    ) as response, open(jar_path, 'wb') as out_file:
        out_file.write(response.content)

    logging.info(f"Successfully download JDBC jar from {jar_file_url} and saved to {jar_path}")
    return jar_path


def download_jdbc_jars(jdbc_driver_folder):
    jars = [
        {
            'db_provider' : 'db2',
            'jar_file_url' : 'https://repo1.maven.org/maven2/com/ibm/db2/jcc/db2jcc/db2jcc4/db2jcc-db2jcc4.jar',
            'jar_file_name' : 'db2jcc-db2jcc4.jar'
        },
        {
            'db_provider' : 'mysql',
            'jar_file_url' : 'https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.27/mysql-connector-java-8.0.27.jar',
            'jar_file_name' : 'mysql-connector-java.jar'
        },
        {
            'db_provider' : 'sqlite',
            'jar_file_url' : 'https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/3.36.0.3/sqlite-jdbc-3.36.0.3.jar',
            'jar_file_name' : 'sqlite-jdbc.jar'
        },
        {
            'db_provider' : 'sqlserver',
            'jar_file_url' : 'https://repo1.maven.org/maven2/com/microsoft/sqlserver/mssql-jdbc/8.4.0.jre8/mssql-jdbc-8.4.0.jre8.jar',
            'jar_file_name' : 'mssql-jdbc-8.4.0.jre8.jar'
        },
        {
            'db_provider' : 'oracle',
            'jar_file_url' : 'https://repo1.maven.org/maven2/com/oracle/database/jdbc/ojdbc8/19.3.0.0/ojdbc8-19.3.0.0.jar',
            'jar_file_name' : 'ojdbc.jar'
        },
        {
            'db_provider' : 'presto',
            'jar_file_url' : 'https://repo1.maven.org/maven2/io/prestosql/presto-jdbc/347/presto-jdbc-347.jar',
            'jar_file_name' : 'presto.jar'
        },
        {
            'db_provider' : 'postgresql',
            'jar_file_url' : 'https://repo1.maven.org/maven2/org/postgresql/postgresql/42.2.18/postgresql-42.2.18.jar',
            'jar_file_name' : 'postgresql.jar'
        },
        {
            'db_provider': 'sas',
            'jar_file_url': 'https://repos.spark-packages.org/saurfang/spark-sas7bdat/3.0.0-s_2.12/spark-sas7bdat-3.0.0-s_2.12.jar',
            'jar_file_name': 'spark-sas.jar'
        },
        {
            'db_provider': 'parso',  # use to read sas dataset
            'jar_file_url': 'https://repo1.maven.org/maven2/com/epam/parso/2.0.14/parso-2.0.14.jar',
            'jar_file_name': 'parso.jar'
        }
    ]

    if not os.path.exists(jdbc_driver_folder):
        os.makedirs(jdbc_driver_folder)

    for jar in jars:
        db_provider = jar.get('db_provider')
        jar_file_url = jar.get('jar_file_url')
        jar_file_name = jar.get('jar_file_name')

        download_jdbc_jar(jar_file_url, jdbc_driver_folder, jar_file_name)


def readExcel(file, sheetName: str = None, dtype = "str", na_filter: bool = True):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        try:
            with pd.ExcelFile(file) as reader:
                try:
                    file = pd.read_excel(reader, sheetname = sheetName, engine = "python", encoding = "utf8", dtype = dtype, na_filter = na_filter )
                except:
                    file = pd.read_excel(reader, sheet_name = sheetName, dtype = dtype, na_filter = na_filter)
        except Exception as e:
            logging.critical(e)
            file = pd.DataFrame()
    return file


def compare_prod_test(prod_df: pd.DataFrame, test_df: pd.DataFrame):
    """Compare prod dataframe and test dataframe on missing value and matching rate by feature

    :param prod_df: prod version df, benchmark, make sure the joining index is set (single or multiindex)
    :param test_df: test version df, challenger, make sure the joining index is set (single or multiindex)
    :return:
    """
    features = pd.Index(prod_df.columns).intersection(test_df.columns)
    print(f"Size Check:")
    print(f"\tprod_df: {prod_df.shape}  vs. test_df: {test_df.shape}")
    print(f"\t{len(features)} common features found: {list(features)}")
    compare_df = pd.merge(
        prod_df[features],
        test_df[features],
        how='inner',
        left_index=True,
        right_index=True,
        suffixes=('_PROD', '_TEST')
    )
    print(
        f"\t{len(compare_df)} common records found, match rate: prod_df {len(compare_df) / len(prod_df) :.1%}, test_df {len(compare_df) / len(test_df) :.1%}")

    print("\nMatch Rate By Feature")
    # 1.Numerical Feature Check
    numeric_compare = compare_df.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])
    numeric_features = numeric_compare.columns.str.replace('_PROD', '').str.replace('_TEST', '').drop_duplicates(
        keep='first')

    # 2.Categorical Feature Check
    categ_compare = compare_df.select_dtypes(include=['object'])
    categ_features = categ_compare.columns.str.replace('_PROD', '').str.replace('_TEST', '').drop_duplicates(
        keep='first')

    # 3.Datetime Feature Check
    dt_compare = compare_df.select_dtypes(include=['datetime64'])
    dt_features = dt_compare.columns.str.replace('_PROD', '').str.replace('_TEST', '').drop_duplicates(keep='first')

    matching = pd.DataFrame(index=features, columns=['DTYPE', 'NA_PROD', 'NA_TEST', 'MATCH', 'MATCH_VALID'])
    unmatching = {}

    for f in tqdm(numeric_features):
        matching.loc[f, 'DTYPE'] = 'NUM'
        matching.loc[f, 'MATCH'] = np.isclose(compare_df[f + "_PROD"], compare_df[f + "_TEST"]).sum() / len(compare_df)
        matching.loc[f, 'NA_PROD'] = compare_df[f + "_PROD"].isnull().sum() / len(compare_df)
        matching.loc[f, 'NA_TEST'] = compare_df[f + "_TEST"].isnull().sum() / len(compare_df)
        # DROP NA MATCH
        compare_df_valid = compare_df.dropna(subset = [f + "_PROD", f + "_TEST"])
        matching.loc[f, 'MATCH_VALID'] = np.isclose(compare_df_valid[f + "_PROD"], compare_df_valid[f + "_TEST"]).sum() / len(compare_df_valid)

        unmatching[f] = compare_df.loc[
            ~np.isclose(compare_df[f + "_PROD"], compare_df[f + "_TEST"]), [f + "_PROD", f + "_TEST"]]

    for f in tqdm(categ_features):
        matching.loc[f, 'DTYPE'] = 'CATEG'
        matching.loc[f, 'MATCH'] = (compare_df[f + "_PROD"] == compare_df[f + "_TEST"]).sum() / len(compare_df)
        matching.loc[f, 'NA_PROD'] = compare_df[f + "_PROD"].isnull().sum() / len(compare_df)
        matching.loc[f, 'NA_TEST'] = compare_df[f + "_TEST"].isnull().sum() / len(compare_df)
        # DROP NA MATCH
        compare_df_valid = compare_df.dropna(subset=[f + "_PROD", f + "_TEST"])
        matching.loc[f, 'MATCH_VALID'] = (compare_df_valid[f + "_PROD"] == compare_df_valid[f + "_TEST"]).sum() / len(compare_df_valid)

        unmatching[f] = compare_df.loc[compare_df[f + "_PROD"] != compare_df[f + "_TEST"], [f + "_PROD", f + "_TEST"]]

    for f in tqdm(dt_features):
        matching.loc[f, 'DTYPE'] = 'CATEG'
        matching.loc[f, 'MATCH'] = (pd.to_datetime(compare_df[f + "_PROD"]) == pd.to_datetime(
            compare_df[f + "_TEST"])).sum() / len(compare_df)
        matching.loc[f, 'NA_PROD'] = compare_df[f + "_PROD"].isnull().sum() / len(compare_df)
        matching.loc[f, 'NA_TEST'] = compare_df[f + "_TEST"].isnull().sum() / len(compare_df)
        # DROP NA MATCH
        compare_df_valid = compare_df.dropna(subset=[f + "_PROD", f + "_TEST"])
        matching.loc[f, 'MATCH_VALID'] = (pd.to_datetime(compare_df_valid[f + "_PROD"]) == pd.to_datetime(
            compare_df_valid[f + "_TEST"])).sum() / len(compare_df_valid)

        unmatching[f] = compare_df.loc[
            pd.to_datetime(compare_df[f + "_PROD"]) != pd.to_datetime(compare_df[f + "_PROD"]), [f + "_PROD",
                                                                                                 f + "_TEST"]]

    matching.sort_values(by=['MATCH'], ascending=[False], inplace=True)
    return matching, unmatching

def error_analysis(unmatching_f: pd.DataFrame, dtype: str):
    """

    :param unmatching_f:
    :param dtype:{INT, FLOAT, DATE, CATEG}
    :return:
    """

    un = unmatching_f.copy()
    if dtype == 'INT':
        un['DIFF'] = un.iloc[:, 0] - un.iloc[:, 1]
        return un['DIFF'].value_counts(normalize = True, dropna = False)
    if dtype == 'FLOAT':
        un['DIFF'] = un.iloc[:, 0] - un.iloc[:, 1]
        return un.describe()
    if dtype == 'CATEG':
        from sklearn.metrics import confusion_matrix

        value_categs = un.iloc[:, 0].value_counts().index
        return pd.DataFrame(
            confusion_matrix(un.iloc[:, 0], un.iloc[:, 1], labels = value_categs),
            columns = value_categs,
            index = value_categs
        ).style.applymap(lambda x: "background-color: #F28C77" if x > 0 else "background-color: white")

def loop_timer(total_steps: int) -> Tuple[int, float, float, float, float]:
    _start = time.time()
    for step in range(total_steps):
        _lap = time.time()
        progress = (step + 1) / total_steps
        time_step = _lap - _start
        time_elasped = _lap - _start
        time_remaining = time_elasped / (step + 1) * (total_steps - step - 1)
        yield step + 1, progress, time_step, time_elasped, time_remaining


def convert_np_dtype(df: pd.DataFrame, exclude_cols: List[str] = None, keep_nonnull_int: bool = True, keep_date:bool = True) -> pd.DataFrame:
    """convert dtype from pandas to numpy compatible (sklearn), i.e., only object, datetime, int and float type

    :param df:
    :param exclude_cols: columns to exclude from conversion
    :param keep_nonnull_int: if True, will convert non-null nullable Integer to numpy non-nullable integer,
            otherwise will convert nullable Integer to float
    :param keep_date: if False, will convert datetime type to date string
    :return:
    """

    if exclude_cols is None:
        exclude_cols = []
    # convert nullable integer
    dtypes = df.dtypes
    nullable_int_cols = dtypes[
        (dtypes.apply(lambda d: d.name).str.startswith("Int"))
        | (dtypes.apply(lambda d: d.name).str.startswith("UInt"))
        ].index

    non_null_int_cols = []
    if keep_nonnull_int:
        # will convert non-null nullable Integer to numpy non-nullable integer
        null_int_stat = df[nullable_int_cols].isna().sum()
        non_null_int_cols = null_int_stat[null_int_stat == 0].index
        nullable_int_cols = nullable_int_cols.difference(non_null_int_cols)

    if not keep_date:
        for col in df.select_dtypes(include=['datetime']):
            df[col] = df[col].dt.date.apply(dt2str)

    return df.astype(
        {
            c: 'object'
            for c in df.select_dtypes(exclude=['number', 'datetime']).columns
            if c not in exclude_cols
        }
    ).astype(
        {
            c: 'float'
            for c in nullable_int_cols
            if c not in exclude_cols
        }
    ).astype(
        {
            c: 'int'
            for c in non_null_int_cols
            if c not in exclude_cols
        }
    )


class TempFolder:
    def __init__(self, root_path: str):
        folder = str(uuid.uuid4())
        self.temp_folder = os.path.join(root_path, folder)

    def __enter__(self):
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.temp_folder, ignore_errors = True)

    def getTempFolderPath(self):
        return self.temp_folder

def save_ibis_to_parquet_on_fs(df: ibis.expr.types.Table, fs: AbstractFileSystem, filepath: str, **kws):
    """save ibis dataframe (from any backend) to parquet file on given filesystem

    :param ibis.expr.types.Table df: ibis dataframe
    :param AbstractFileSystem fs: fsspec compatible filesystem
    :param str filepath: root path of file, if on object storage, 
            the full path including buckets
    """
    chunk_size = kws.get('chunk_size', 1048576)
    with fs.open(path = filepath, mode = 'wb') as obj:
        df_arr: pa.RecordBatchReader = df.to_pyarrow_batches(chunk_size = chunk_size)
        with pq.ParquetWriter(where=obj, schema=df_arr.schema, **kws) as writer:
            for df_batch in df_arr:
                writer.write_batch(df_batch)
                
def save_pandas_to_parquet_on_fs(df: pd.DataFrame, fs: AbstractFileSystem, filepath: str, **kws):
    """save pandas dataframe (from any backend) to parquet file on given filesystem

    :param pd.DataFrame df: pandas dataframe
    :param AbstractFileSystem fs: fsspec compatible filesystem
    :param str filepath: root path of file, if on object storage, 
            the full path including buckets
    """
    with fs.open(path = filepath, mode = 'wb') as obj:
        df_arr: pa.Table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(where=obj, schema=df_arr.schema, **kws) as writer:
            writer.write_table(df_arr)


if __name__ == '__main__':
    dt = date(2021, 1, 4)
    print(dt2str(dt))


