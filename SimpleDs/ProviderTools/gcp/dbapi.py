import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from CommonTools.dbapi import baseDbInf

class BigQuery(baseDbInf):
    def __init__(self, project_id: str, driver="bigquery", credentials_path: str = None):
        """
        bq = BigQuery(project_id='your-project-id')
        bq.bindServer(db = 'your-dataset-id')
        bq.launch()
        """
        super().__init__(driver)
        self.project_id = project_id
        self.credentials_path = credentials_path # json file
        self.bindServer(db = None)

    def argTempStr(self):
        return "%s"

    def getJdbcUrl(self) -> str:
        return f"jdbc:bigquery://{self.ip}:{self.port};ProjectId={self.project_id}"

    def getDriverClass(self) -> str:
        # https://mvnrepository.com/artifact/com.google.cloud/google-cloud-bigquery/2.37.1
        return "com.simba.googlebigquery.jdbc42.Driver"

    def getConnStr(self) -> str:
        # only used for sqlalchemy driver type
        return f"{self.driver}://{self.project_id}/{self.db}"
    
    def bindServer(self, ip: str = 'https://www.googleapis.com/bigquery/v2', port: int = 443, db: str = None):
        """Connect to a database server

        :param ip: ip of the server
        :param port: port number of the server
        :param db: which database to connect
        :return:
        """
        self.ip = ip
        self.port = port
        self.db = db

    def launch(self):
        """Launch the databse connector, create the sqlalchemy engine and create a session

        :return:
        """
        connStr = self.getConnStr()
        self.engine = create_engine(connStr, credentials_path=self.credentials_path)
        self.DBSession = sessionmaker(bind = self.engine)
        logging.info("Engine started, ready to go!")