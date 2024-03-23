import logging
import os
import socket
from pyspark.sql import SparkSession

from luntaiDs.CommonTools.data_structure import EnhancedDict
from luntaiDs.CommonTools.dbapi import baseDbInf


class SparkConnector:

    @classmethod
    def setup(cls, jdbc_jars: list = None, s3_config : dict = None, **additional_params: dict):
        """set up for the environment to run successfully

        :param jdbc_jars: if you want to connect to database (MySQL, SQL server, Hive, Teradata, etc) set this. ['path_to_jar1', 'path_to_jar2]
        :param s3_config: if you want to connect to AWS S3, set s3_configuration to connect, need to provide: {'STORAGE_URL', 'ACCESS_KEY', 'SECRET_ACCESS_KEY', 'SESSION_TOKEN'}
        :param additional_params: additional kws to pass to spark config
        :return:
        """

        cls.JAR_PATHS = ":".join(jdbc_jars)
        logging.info(f"JDBC Driver Jars: {cls.JAR_PATHS}")

        ## build the configs
        cls.conf = EnhancedDict()
        cls.conf["spark.sql.execution.arrow.pyspark.enabled"] = True  # fast transform from pandas to spark
        cls.conf["spark.executor.allowSparkContext"] = True # https://learn.microsoft.com/en-us/answers/questions/941566/sparkcontext-should-only-be-created-and-accessed-o.html

        if s3_config:
            cls.conf["spark.hadoop.fs.s3a.endpoint"] = s3_config.get("STORAGE_URL")
            cls.conf["spark.hadoop.fs.s3a.aws.credentials.provider"] = 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider'
            cls.conf["spark.hadoop.fs.s3a.access.key"] = s3_config.get("ACCESS_KEY")
            cls.conf["spark.hadoop.fs.s3a.secret.key"] = s3_config.get("SECRET_ACCESS_KEY")
            cls.conf["spark.hadoop.fs.s3a.session.token"] = s3_config.get("SESSION_TOKEN")
            cls.conf["spark.hadoop.fs.s3a.fast.upload"] = True
            cls.conf["spark.hadoop.fs.s3a.path.style.access"] = True
            cls.conf["spark.hadoop.fs.s3a.impl"] = "org.apache.hadoop.fs.s3a.S3AFileSystem"
            cls.conf["spark.hadoop.fs.s3a.multipart.size"] = "128M"
            cls.conf["spark.hadoop.fs.s3a.fast.upload.active.blocks"] = 8
        if jdbc_jars:
            cls.conf["spark.driver.extraClassPath"] = cls.JAR_PATHS

        for k, v in additional_params.items():
            cls.conf[k] = v


    @classmethod
    def getSparkSession(cls, app_name: str = "SparkConnector", master: str = "local[*]"):

        sparkSessionBuilder = SparkSession.builder.master(master).appName(app_name)

        for k, v in cls.conf.items():
            sparkSessionBuilder = sparkSessionBuilder.config(k, v)

        spark = sparkSessionBuilder.getOrCreate()

        return spark

    def __init__(self, app_name: str = "SparkConnector", master: str = "local"):
        if not hasattr(self, 'JAR_PATHS'):
            raise Exception("Must call SparkConnector.setup() before initialization")

        self.spark = self.getSparkSession(app_name, master)
        logging.info(
            f"Spark session object successfully created, call SparkConnector({app_name}).spark to get spark obj")

    def get_file_path(self, path: str) -> str:
        return path

    def get_common_reader(self, header=True, sep=",", schema=None):
        if schema:
            return self.spark.read.option('header', header) \
                .option("inferSchema", "false") \
                .option("sep", sep) \
                .option("encoding", "utf-8") \
                .schema(schema)
        else:
            return self.spark.read.option('header', header) \
                .option("inferSchema", "true") \
                .option("sep", sep) \
                .option("encoding", "utf-8")

    def read_csv(self, *path: str, header=True, sep=",", schema=None):
        """read csv file(s) into spark

        :param path: can either be one single csv file, or multiple files as positional arguments
        :param header: whether it has a header
        :param sep: what separator to use
        :param schema: if you have a spark schema, attach it
        :return: spark dataframe

        >>> # for the `path` argument, can read one single file as:
        >>> df = sc.read_csv(path, sep = ',')
        >>> # or read multiple csv files with same schema (usually in one folder)
        >>> df = sc.read_csv(path1, path2, path3, sep = ',')
        """
        # path = self.get_file_path(path)

        try:
            df = self.get_common_reader(header, sep, schema).csv(*path)
        except Exception as e:
            logging.error('File Load FAILED: ' + str(e))
        else:
            logging.info('File Load successfully.')
            return df

    def read_sas(self, path: str):
        # path = self.get_file_path(path)

        try:
            df = self.spark.read.format("com.github.saurfang.sas.spark").load(path)
        except Exception as e:
            logging.error('File Load FAILED: ' + str(e))
        else:
            logging.info('File Load successfully.')
            return df

    def read_parquet(self, *path: str, header=True, sep=",", schema=None):
        """read parquet file(s) into spark

        :param path: can either be one single csv file, or multiple files as positional arguments
        :param header: whether it has a header
        :param sep: what separator to use
        :param schema: if you have a spark schema, attach it
        :return: spark dataframe

        >>> # for the `path` argument, can read one single file as:
        >>> df = sc.read_parquet(path, sep = ',')
        >>> # or read multiple csv files with same schema (usually in one folder)
        >>> df = sc.read_parquet(path1, path2, path3, sep = ',')
        """
        # path = self.get_file_path(path)

        try:
            df = self.get_common_reader(header, sep, schema).parquet(*path)
        except Exception as e:
            logging.error('File Load FAILED: ' + str(e))
        else:
            logging.info('File Load successfully.')
            return df

    def save_csv(self, df, path: str, header=True, sep=",", mode="overwrite", repartition: int = None):
        """save spark dataframe to csv file (partitions saved in sub folder)

        :param df: spark dataframe
        :param path: path of the csv file (actually a folder named xxx.csv with partitions inside)
        :param header: whether to include a header
        :param sep:
        :param mode: {overwrite or append or ignore}
        :param repartition: number of partitions to split the dataframe into
        :return:
        """
        path = self.get_file_path(path)

        if repartition:
            df = df.repartition(repartition)

        try:
            df.write.option('header', header).option("sep", sep).mode(mode).csv(path)
        except Exception as e:
            logging.error('File Wrote FAILED: ' + str(e))
        else:
            logging.info('File Wrote successfully.')

    def save_parquet(self, df, path: str, header=True, sep=",", mode="overwrite", repartition: int = None):
        """save spark dataframe to parquet file (partitions saved in sub folder)

        :param df: spark dataframe
        :param path: path of the parquet file (actually a folder named xxx.csv with partitions inside)
        :param header: whether to include a header
        :param sep:
        :param mode: {overwrite or append or ignore}
        :param repartition: number of partitions to split the dataframe into
        :return:
        """
        path = self.get_file_path(path)

        if repartition:
            df = df.repartition(repartition)

        try:
            df.write.option('header', header).option("sep", sep).mode(mode).parquet(path)
        except Exception as e:
            logging.error('File Wrote FAILED: ' + str(e))
        else:
            logging.info('File Wrote successfully.')

    def query_db(self, dbConf: baseDbInf, sql: str, result_name: str = "temp_df", **kws):
        """query_extract from database and return spark dataframe

        :param dbConf: the dbConf object for connection string setup
        :param sql: the sql query_extract to make
        :param result_name: spark jdbc requires each query_extract result to have a table name
        :param kws: other parameters that could parse into .jdbc(properties)
        :return: spark DataFrame
        """

        sql = f"({sql.replace(';', '')}) {result_name}"  # drop ; and wrap with temp table name
        properties = {
            "driver": dbConf.getDriverClass(),
            # "dbtable" : sql,
        }

        if hasattr(dbConf, 'username'):
            properties["user"] = dbConf.username
            properties["password"] = dbConf.password

        for k, v in kws.items():
            properties[k] = v

        try:
            df = self.spark.read.jdbc(
                url=dbConf.getJdbcUrl(),
                table=sql,
                properties=properties
            )
        except Exception as e:
            logging.error('Query Load FAILED: ' + str(e))
        else:
            logging.info('Query Load successfully.')
            return df

    def save_db(self, dbConf: baseDbInf, df, table_name: str, mode: str = None, column_schema: dict = None,
                **kws) -> int:
        """save df to database

        :param dbConf: the dbConf object for connection string setup
        :param df: spark dataframe to save
        :param table_name: name of the table in the database
        :param mode: {None, append, overwrite}, None will fail if table exists, append will append to end, overwrite will discard existing values
        :param column_schema: dictionary of column and dtype, e.g. column_schema = {"cur" : "varchar(4)", "v1": "float"}
        :param kws: other parameters that could parse into .jdbc(properties)
        :return: 1 for success and 0 for failed
        """
        properties = {
            "driver": dbConf.getDriverClass(),
            "dbtable": table_name,
        }

        if hasattr(dbConf, 'username'):
            properties["user"] = dbConf.username
            properties["password"] = dbConf.password

        if column_schema:
            col_sch = ",".join(f"{k} {v}" for k, v in column_schema.items())
            properties["createTableColumnTypes"] = col_sch

        for k, v in kws.items():
            properties[k] = v

        try:
            df.write.jdbc(
                url=dbConf.getJdbcUrl(),
                table=table_name,
                mode=mode,
                properties=properties
            )
        except Exception as e:
            logging.error('Save to DB FAILED: ' + str(e))
            return 0
        else:
            logging.info('Save to DB successfully.')
            return 1


class SparkConnectorK8s(SparkConnector):
    @classmethod
    def setup(cls, jdbc_jars: list = None, s3_config : dict = None):
        """set up for the environment to urn successfully

        :param jdbc_jars: if you want to connect to database (MySQL, SQL server, Hive, Teradata, etc) set this. ['path_to_jar1', 'path_to_jar2]
        :param s3_config: if you want to connect to AWS S3, set s3_configuration to connect, need to provide: {'STORAGE_URL', 'ACCESS_KEY', 'SECRET_ACCESS_KEY', 'SESSION_TOKEN'}
        :return:
        """

        cls.JAR_PATHS = ":".join(jdbc_jars)
        logging.info(f"JDBC Driver Jars: {cls.JAR_PATHS}")

        os.environ['SPARK_CONF_DIR'] = os.environ["EDL_CONF_LOCATION"]
        krb_conf = os.getenv('KRB5_CONFIG')
        os.environ["SPARK_SUBMIT_OPTS"] = f"-Djava.security.krb5.conf={krb_conf}"
        SID = os.getenv("JUPYTERHUB_USER")

        os.environ['PYSPARK_PYTHON'] = 'python3.8'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.8'

        cls.conf = EnhancedDict()

        cls.conf["spark.kubernetes.container.image"] = "image-path"
        cls.conf["spark.kubernetes.authenticate.caCertFile"] = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        cls.conf["spark.kubernetes.authenticate.oauthTokenFile"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        cls.conf["spark.executor.instances"] = "2"
        cls.conf["spark.executor.cores"] = "2"
        cls.conf["spark.kubernetes.executor.request.cores"] = "2"
        cls.conf["spark.kubernetes.executor.limit.cores"] = "2"
        cls.conf["spark.driver.memory"] = "4G"
        cls.conf["spark.driver.cores"] = "2"
        cls.conf["spark.executor.memory"] = "4G"

        cls.conf["spark.driver.host"] = socket.gethostbyname(socket.gethostname())
        cls.conf["spark.kubernetes.namespace"] = os.getenv("POD_NAMESPACE")
        cls.conf["spark.kubernetes.container.image.pullSecrets"] = "artifactory-regcred"
        cls.conf["spark.kubernetes.container.image.pullPolicy"] = "IfNotPresent"
        cls.conf["spark.submit.deployMode"] = "client"
        cls.conf["spark.kubernetes.authenticate.driver.serviceAccountName"] = "spark-user"
        cls.conf["spark.kubernetes.executor.podNamePrefix"] = SID
        ######
        cls.conf["spark.hadoop.fs.s3a.endpoint"] = s3_config.get("STORAGE_URL")
        cls.conf[
            "spark.hadoop.fs.s3a.aws.credentials.provider"] = 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider'
        cls.conf["spark.hadoop.fs.s3a.access.key"] = s3_config.get("ACCESS_KEY")
        cls.conf["spark.hadoop.fs.s3a.secret.key"] = s3_config.get("SECRET_ACCESS_KEY")
        cls.conf["spark.hadoop.fs.s3a.session.token"] = s3_config.get("SESSION_TOKEN")
        cls.conf["spark.hadoop.fs.s3a.fast.upload"] = True
        cls.conf["spark.hadoop.fs.s3a.path.style.access"] = True
        cls.conf["spark.hadoop.fs.s3a.impl"] = "org.apache.hadoop.fs.s3a.S3AFileSystem"
        cls.conf["spark.hadoop.fs.s3a.multipart.size"] = "128M"
        cls.conf["spark.hadoop.fs.s3a.fast.upload.active.blocks"] = 8
        cls.conf["spark.driver.extraClassPath"] = cls.JAR_PATHS

        ######################
        cls.conf["spark.kerberos.principal"] = f"{SID}@server-path"
        cls.conf["spark.kerberos.keytab"] = "/home/jovyan/.keytabs/edl.keytab"
        cls.conf["spark.hadoop.javax.jdo.option.ConnectionUserName"] = SID
        cls.conf["spark.hadoop.dfs.namenode.kerberos.principal.pattern"] = "*"
        cls.conf["spark.sql.hive.metastore.version"] = "1.2.1"
        cls.conf["spark.sql.hive.metastore.jars"] = "local:///" + os.environ["EDL_JARS_LOCATION"]
        cls.conf["spark.sql.execution.arrow.pyspark.enabled"] = "true"

        # cls.conf["spark.driver.extraClassPath"] = cls.JAR_PATHS
        # cls.conf["spark.executor.extraClassPath"] = os.environ["EDL_JARS_LOCATION"]
        # cls.conf["spark.executor.extraClassPath"] = os.environ["EDL_CONF_LOCATION"] + "*"

    @classmethod
    def getSparkSession(cls, app_name: str = "SparkConnectorEDL", master: str = None):
        master = "k8s://https://kubernetes.default.svc.cluster.local:443"
        sparkSessionBuilder = SparkSession.builder.master(master).appName(app_name)

        for k, v in cls.conf.items():
            sparkSessionBuilder = sparkSessionBuilder.config(k, v)

        spark = sparkSessionBuilder.enableHiveSupport().getOrCreate()

        return spark