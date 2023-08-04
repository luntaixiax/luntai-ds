import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

try:
    from pyspark.sql.functions import isnan, when, count, col
    from pyspark.sql import DataFrame
    from pyspark.sql.types import FloatType
    import pyspark.sql.functions as F
except:
    pass

from CommonTools.dbapi import baseDbInf, dbIO


############## categ and numeric column attributer #####################

class CategCol:
    coltype = 'Category'

    def __init__(self, total: int, missing: int, v_c: pd.Series):
        self.total = total
        self.missing = missing
        self.missing_rate = missing / total

        self.v_c = v_c
        self.v_c_rate = v_c / v_c.sum()
        self.num_valid_lvl = len(v_c) # number of different valid non-null value levels

    def __str__(self):
        return f"CategCol([total/missing/rate] = [{self.total}/{self.missing}/{self.missing_rate :.1%}], categ lvls = {self.num_valid_lvl})"

    def get_display_vc(self):
        """only display top 10 most frequent levels and pack the rest into the 'Other' category
        """
        v_c_display = self.v_c.iloc[:10]
        if len(self.v_c.iloc[10:]) > 0:
            v_c_display = v_c_display.append(pd.Series(self.v_c.iloc[10:].sum(), index=['{Other}']))
        return v_c_display

class NumCol:
    coltype = 'Numeric'

    def __init__(self, vector: pd.Series, bins: int = 10):
        self.bins = bins
        self.total = len(vector)
        self.missing = vector.isnull().sum()
        self.missing_rate = self.missing / self.total

        self.vector = vector
        self.v_c = vector.value_counts(dropna = True, normalize = False, bins = bins, sort = False)
        self.v_c_rate = vector.value_counts(dropna = True, normalize = True, bins = bins, sort = False)
        self.quantile = vector.sort_values().quantile(np.linspace(0, 1, bins + 1))

    def __str__(self):
        l, h = self.get_outlier_IQR()
        return f"NumCol([total/missing/rate] = [{self.total}/{self.missing}/{self.missing_rate :.1%}], normal_boundary = [{l}, {h}])\n{self.vector.describe()}"

    def get_outlier_IQR(self):
        # outlier using IQR
        q1, q3 = np.percentile(self.vector, [25, 75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        return lower_bound, upper_bound

    def get_num_high_low(self):
        lower_bound, upper_bound = self.get_outlier_IQR()
        num_high = self.vector[self.vector > upper_bound].count()
        num_low = self.vector[self.vector < lower_bound].count()
        return num_high, num_low


class NumColSpark(NumCol):
    def __init__(self, df: DataFrame, column: str, bins: int = 10):
        try:
            df = df.withColumn(column, df[column].cast(FloatType()))
        except:
            pass

        self.column = column
        self.bins = bins
        self.total = df.count()
        self.missing = df.filter(col(column).isNull()).count()
        self.missing_rate = self.missing / self.total

        self.vector = df.select(column).toPandas()[column.replace("`", "")]
        self.df = df

        breaks, counts = df.select(column).rdd.flatMap(lambda x: x).histogram(bins)  # ([interval points], [counts])
        self.v_c = pd.Series(
            counts,
            index = pd.IntervalIndex.from_breaks(breaks),
            name = column
        )
        self.v_c_rate = self.v_c / self.v_c.sum()

        self.quantile = pd.Series(
            df.approxQuantile(column, list(np.linspace(0, 1, bins + 1)), 0),
            index = np.linspace(0, 1, bins + 1),
            name = column
        )

    def __str__(self):
        l, h = self.get_outlier_IQR()
        des = self.df.select(self.column).summary().toPandas()
        des = pd.Series(
            des[self.column.replace("`", "")].values,
            index = des['summary'].values,
            name = self.column
        )
        return f"NumColSpark([total/missing/rate] = [{self.total}/{self.missing}/{self.missing_rate :.1%}], normal_boundary = [{l}, {h}])\n{des}"

    def get_outlier_IQR(self):
        # outlier using IQR
        q1, q3 = self.df.approxQuantile(self.column, [0.25, 0.75], 0)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        return lower_bound, upper_bound

    def get_num_high_low(self):
        lower_bound, upper_bound = self.get_outlier_IQR()
        num_high = self.df.filter(col(self.column) > upper_bound).count()
        num_low = self.df.filter(col(self.column) < lower_bound).count()
        return num_high, num_low

############## plotter ########################

def _plot_categ(categ_col: CategCol):
    total = categ_col.total
    num_null = categ_col.missing
    v_c_display = categ_col.get_display_vc()

    # add chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    # add subplot for null vs. valid
    axes[0].pie([total - num_null, num_null], labels=['valid', 'missing'],
                autopct = '%.0f%%', explode = (0, 0.05), startangle = 90)
    axes[0].set_title(f"Valid vs. Missing (total {total})")
    # add subplot for valid value counts
    g = sns.barplot(x = v_c_display.index, y = v_c_display, order = v_c_display.index, ax = axes[1])
    g.bar_label(g.containers[0])
    axes[1].tick_params(axis = 'x', rotation = 45)
    axes[1].set_title(f"Valid value counts")

def _plot_num(num_col: NumCol):
    vector = num_col.vector
    total = num_col.total
    num_null = num_col.missing
    num_high, num_low = num_col.get_num_high_low()
    num_valid = total - num_null - num_high - num_low

    # add chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    # add subplot for null vs. valid
    axes[0].pie(
        [num_null, num_valid, num_high, num_low],
        labels = ['missing', 'valid', 'outlier-high', 'outlier-low'],
        autopct = '%.0f%%', explode = (0.05, 0, 0, 0), startangle = 90
    )
    axes[0].set_title(f"Valid vs. Missing (total {total})")
    # add boxplot
    sns.histplot(vector, ax = axes[1], bins = 20)
    axes[1].set_title(f"Valid value distribution")


############## table explorer ########################

class Previewer:
    def view_categ(self, column: str):
        """view categorical univariate feature distribution (missing values, value counts)

        :param column: column name
        :return: statistics of this column
        """
        raise NotImplementedError()

    def view_numeric(self, column: str):
        """view numerical univariate feature distribution (missing value, outlier, histogram)

        :param column: column name
        :return: statistics of this column
        """
        raise NotImplementedError()

class PdPreviewer(Previewer):
    def __init__(self, df: pd.DataFrame):
        """Previewer for pandas dataframe

        :param df: pandas dataframe
        """
        self.df = df

    def view_categ(self, column: str):
        vector = self.df[column]
        total = len(vector)
        num_null = vector.isnull().sum()
        v_c = vector.value_counts(dropna = True, normalize = False, sort = True)

        categ_col = CategCol(total, num_null, v_c)
        _plot_categ(categ_col)
        return categ_col

    def view_numeric(self, column: str, bins: int = 10):
        vector = self.df[column]

        num_col = NumCol(vector, bins)
        _plot_num(num_col)
        return num_col


class DbPreviewer(Previewer):
    def __init__(self, db_conf: baseDbInf, tb: str = None, sql: str = None, view_schema : str = None):
        """previewer for database

        :param db_conf: database conf object
        :param tb: optional, if you need do EDA on the entire table, specify table name here (schema.table)
        :param sql: optional, if you only want to EDA on partial table or transformed table, specify sql query_extract statment
        :param view_schema: please specify the schema when you specify `sql` param, this will create a temporary view under the schema provided
        """
        self.db = dbIO(db_conf)
        if sql:
            # this is the view to show
            if view_schema:
                self.tb = f"{view_schema}.aq_temp_db_previewer"
            else:
                self.tb = "aq_temp_db_previewer"
            # delete the existing view
            sql_t = f"Drop view {self.tb}"
            r = self.db.modify_sql(sql_t, errormsg = f"Drop view Failed: {self.tb}")
            # create a view into the database
            sql = f"Create view {self.tb} as\n" + sql.replace(";", "")
            r = self.db.modify_sql(sql, errormsg = f"Create view Failed: {self.tb}")
            if r == 0:
                raise ValueError("Fail to create the view, please try adding the view directly into database yourself")
        else:
            if tb:
                self.tb = tb
            else:
                raise ValueError("At least table name (tb) and sql query_extract (sql) should be specified")

    def view_categ(self, column: str):
        column_t = f"`{column}`" if "." in column else column
        # db2 col name need to be capitalized
        total = self.db.query_sql_df(f"select count(*) from {self.tb}").iloc[0,0]
        num_null = self.db.query_sql_df(f"select count(*) from {self.tb} where {column_t} is null").iloc[0,0]
        v_c = self.db.query_sql_df(f"select {column_t}, count({column_t}) as NUM from {self.tb} where {column_t} is not null group by {column_t} order by NUM desc").set_index(column)
        v_c = pd.Series(v_c["NUM"].values, index = v_c.index, name = column)

        categ_col = CategCol(total, num_null, v_c)
        _plot_categ(categ_col)
        return categ_col


    def view_numeric(self, column: str, bins: int = 10):
        column_t = f"`{column}`" if "." in column else column
        vector = self.db.query_sql_df(f"select {column_t} from {self.tb}")
        vector = pd.Series(vector[column].values, index = vector.index, name = column)

        num_col = NumCol(vector, bins)
        _plot_num(num_col)
        return num_col

    def reset(self):
        # delete views created
        sql_t = f"Drop view {self.tb}"
        self.db.modify_sql(sql_t, errormsg = f"Drop view Failed: {self.tb}")




class SparkPreviewer(Previewer):
    def __init__(self, df: DataFrame):
        """previewer for pyspark dataframe

        :param df: pyspark dataframe
        """
        self.df = df

    def view_categ(self, column: str):
        column = f"`{column}`" if "." in column else column
        total = self.df.count()
        num_null = self.df.filter(col(column).isNull()).count()
        v_c = self.df.filter(col(column).isNotNull()).groupBy(column).count().orderBy('count', ascending = False).toPandas()
        v_c = pd.Series(v_c['count'].values, index = v_c[column], name = column)

        categ_col = CategCol(total, num_null, v_c)
        _plot_categ(categ_col)
        return categ_col

    def view_numeric(self, column: str, bins: int = 10):
        column = f"`{column}`" if "." in column else column

        num_col = NumColSpark(self.df, column)
        _plot_num(num_col)
        return num_col