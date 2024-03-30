from typing import List, Dict, Union
import ibis
import pandas as pd
import logging
from luntaiDs.ModelingTools.Serving.data_registry import _BaseModelDataRegistry
from luntaiDs.ProviderTools.gcp.dbapi import WarehouseHandlerBQSQL


class _BaseModelDataRegistryBQ(_BaseModelDataRegistry):
    DATA_ID_COL = "DATA_ID_"
    TRAIN_TEST_IND_COL = "IS_TRAIN_"
    
    def __init__(self, handler: WarehouseHandlerBQSQL, schema: str, table: str):
        """initialize

        :param WarehouseHandlerBQSQL handler: bigquery handler
        :param str schema: schema for the table storing modeling data
        :param str table: table storing modeling data
        """
        self.handler = handler
        self.schema = schema
        self.table = table
        
    def init_table(self):
        raise NotImplementedError("")
    
    def get_table(self) -> ibis.expr.types.Table:
        """get underlying training table using ibis

        :return ibis.expr.types.Table: ibis table
        """
        return self.handler.get_table(schema = self.schema, table = self.table)

    def get_existing_ids(self) -> List[str]:
        """get list of registered data ids

        :return List[str]: data ids
        """
        return (
            self.get_table()
            .select(self.DATA_ID_COL)
            .distinct()
            .to_pandas()
            [self.DATA_ID_COL]
            .tolist()
        )

    def register(self, data_id: str, train_ds: Union[ibis.expr.types.Table | pd.DataFrame], 
                 test_ds: Union[ibis.expr.types.Table | pd.DataFrame], replace: bool = False):
        """register table in ibis format

        :param str data_id: data id
        :param Union[ibis.expr.types.Table | pd.DataFrame] train_ds: training dataset
        :param Union[ibis.expr.types.Table | pd.DataFrame] test_ds: testing dataset
        :param bool replace: whether to replace existing dataset, defaults to False
        """
        if replace:
            self.remove(data_id = data_id)
            
        if isinstance(train_ds, pd.DataFrame) and isinstance(test_ds, pd.DataFrame):
            # add data_id column
            train_ds.loc[:, self.DATA_ID_COL] = data_id
            test_ds.loc[:, self.DATA_ID_COL] = data_id
            # add train test indicator
            train_ds.loc[:, self.TRAIN_TEST_IND_COL] = True
            test_ds.loc[:, self.TRAIN_TEST_IND_COL] = False
            # save to database
            self.handler.save_pandas(
                df = train_ds,
                schema = self.schema,
                table = self.table
            )
            self.handler.save_pandas(
                df = test_ds,
                schema = self.schema,
                table = self.table
            )
        elif isinstance(train_ds, ibis.expr.types.Table) and isinstance(test_ds, ibis.expr.types.Table):
            # add data_id column
            train_ds = (
                train_ds
                .mutate(ibis.literal(data_id, 'String').name(self.DATA_ID_COL))
                .mutate(ibis.literal(True, 'Boolean').name(self.TRAIN_TEST_IND_COL))
            )
            test_ds = (
                train_ds
                .mutate(ibis.literal(data_id, 'String').name(self.DATA_ID_COL))
                .mutate(ibis.literal(False, 'Boolean').name(self.TRAIN_TEST_IND_COL))
            )
            # save training set to database
            query_train = ibis.to_sql(train_ds)
            qry_cols_train = self.handler.query(query_train).columns
            sql_train = f"""
            INSERT {self.schema}.{self.table} ({','.join(qry_cols_train)})
            {query_train}
            """
            logging.info(f"Inserting into {self.schema}.{self.table} using query:\n{sql_train}")
            self.handler.execute(sql_train)
            # save testing set to database
            query_test = ibis.to_sql(test_ds)
            qry_cols_test = self.handler.query(query_test).columns
            sql_test = f"""
            INSERT {self.schema}.{self.table} ({','.join(qry_cols_test)})
            {query_test}
            """
            logging.info(f"Inserting into {self.schema}.{self.table} using query:\n{sql_test}")
            self.handler.execute(query_test)    
            
        else:
            raise ValueError("train_ds and test_ds must be of type either pandas or ibis dataframe")
            

    def fetch(self, data_id: str, target_col: str = None):
        """fetch training/testing dataset

        :param str data_id: data id to be fetched
        :param str target_col: the target column, defaults to None.
        :return:
            - if target_col given, will split to [X_train, y_train, X_test, y_test]
            - if target_col not given, will just split to [train_ds, test_ds]
        """
        table = self.get_table()
        train_ds = (
            table
            .filter((table[self.DATA_ID_COL] == data_id) & (table[self.TRAIN_TEST_IND_COL] == True))
            .drop(self.DATA_ID_COL, self.TRAIN_TEST_IND_COL)
        )
        test_ds = (
            table
            .filter((table[self.DATA_ID_COL] == data_id) & (table[self.TRAIN_TEST_IND_COL] == False))
            .drop(self.DATA_ID_COL, self.TRAIN_TEST_IND_COL)
        )
        if target_col:
            X_train = train_ds.drop(target_col)
            X_test = test_ds.drop(target_col)
            y_train = train_ds[target_col]
            y_test = test_ds[target_col]
            return X_train, y_train, X_test, y_test
        else:
            return train_ds, test_ds

    def remove(self, data_id: str):
        """remove dataset from registry

        :param str data_id: data id to be removed
        """
        sql = f"""
        DELETE {self.schema}.{self.table}
        WHERE {self.DATA_ID_COL} = '{data_id}'
        """
        logging.info(f"Deleting table {self.schema}.{self.table} using query:\n{sql}")
        self.handler.execute(sql)