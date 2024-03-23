import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from luntaiDs.CommonTools.dbapi import baseDbInf


class TeraPlaceholderEngine:
    def __init__(self, engine):
        self.engine = engine

    def connect(self):
        return self.engine

    def raw_connection(self):
        return self.engine

    def begin(self):
        return self.engine

class TeraData(baseDbInf):
    def __init__(self, driver_type:str = "sqlalchemy", driver = "teradatasql", log_mechanism: str = "LDAP"):
        """

        :param driver_type: {odbc, teradata, sqlalchemy}
        :param driver:
        :param log_mechanism: {LDAP, TD2, JWT, KRB5}

        https://pypi.org/project/teradatasqlalchemy/

        """
        super().__init__(driver)
        self.driver_type = driver_type
        self.log_mechanism = log_mechanism

    def argTempStr(self):
        return "?"

    def getConnStr(self) -> str:
        # only used for sqlalchemy driver type
        'teradatasql://{username}:{password}@{hostname}'
        return f"{self.driver}://{self.username}:{self.password}@{self.ip}"

    def launch(self):
        """Launch the databse connector, create the sqlalchemy engine and create a session

        :return:
        """
        if self.driver_type == "odbc":
            import pyodbc

            engine = pyodbc.connect(f'DSN=Teradata;'f'UID={self.username}; PWD={self.password}')
            engine.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
            engine.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")
            engine.setencoding(encoding="utf-8")
            self.engine = TeraPlaceholderEngine(engine)

        elif self.driver_type == "teradata":
            import teradatasql
            
            engine = teradatasql.connect(host = self.ip, user = self.username, password = self.password)
            self.engine = TeraPlaceholderEngine(engine)

        elif self.driver_type == "sqlalchemy":
            connStr = self.getConnStr()
            self.engine = create_engine(connStr)
            self.DBSession = sessionmaker(bind = self.engine)

        logging.info("Engine started, ready to go!")

    def getJdbcUrl(self) -> str:
        # for pyspark use
        return f"jdbc:teradata://{self.ip}/{self.db}"

    def getDriverClass(self) -> str:
        # for pyspark use
        return "com.teradata.jdbc.TeraDriver"