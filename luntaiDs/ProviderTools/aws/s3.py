import io
import os

from botocore.exceptions import ClientError
import logging
import pandas as pd
import boto3
from luntaiDs.CommonTools.obj_storage import ObjStorage, remove_begin_slash, add_tail_slash

class S3Accessor(ObjStorage):
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None,
                 aws_session_token=None, region_name=None,
                 botocore_session=None, profile_name=None,
                 service_name:str = 's3', api_version=None,
                 use_ssl=None, verify=None, endpoint_url=None,
                 config=None
                 ):
        self.region_name = region_name
        try:
            self.session = boto3.Session(
                aws_access_key_id = aws_access_key_id,
                aws_secret_access_key = aws_secret_access_key,
                aws_session_token = aws_session_token,
                region_name = region_name,
                botocore_session = botocore_session,
                profile_name = profile_name
            )

            self.client = self.session.client(
                service_name, api_version = api_version, use_ssl = use_ssl,
                verify = verify, endpoint_url = endpoint_url,
                config = config
            )
            self.resource = self.session.resource(
                service_name, api_version = api_version, use_ssl = use_ssl,
                verify = verify, endpoint_url = endpoint_url,
                config = config
            )

        except Exception as e:
            logging.error(f"Boto3 failed to create client to access {service_name} \n{e}")
        else:
            logging.info("Boto3 session/client/resource created successfully.")
            
    def get_uri_protocol(self) -> str:
        return "s3a"

    def list_buckets(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.client.list_buckets()['Buckets'])

    def create_bucket(self, bucket_name: str) -> int:
        """Create a new bucket

        :param bucket_name: name of the bucket
        :return:
        """
        if self.region_name:
            region_config = {'LocationConstraint': self.region_name}
        else:
            region_config = None
        self.client.create_bucket(
            Bucket = bucket_name,
            CreateBucketConfiguration = region_config
        )
        return 1

    def get_bucket(self, bucket_name: str = None):
        return self.resource.Bucket(bucket_name)

    def get_buckets(self):
        return self.resource.buckets

    def get_bucket_iter(self):
        for bucket in self.resource.buckets:
            yield bucket

    def enter_bucket(self, bucket_name: str = None) -> int:
        """move to a given bucket

        :param bucket_name: name of the bucket you want to switch to
        :return:
        """
        # check if exists
        if self.resource.Bucket(bucket_name) in self.resource.buckets.all():
            self.BUCKET = bucket_name
            return 1
        else:
            logging.error(f"Bucket {bucket_name} does not exists, please call .list_buckets() to see all available buckets")
            return 0

    def if_not_exist_return_default_bucket(self, bucket_name = None):
        return self.BUCKET if bucket_name is None else bucket_name


    def get_current_bucket(self):
        """get your current bucket object

        :return:
        """
        if hasattr(self, 'BUCKET'):
            return self.resource.Bucket(self.BUCKET)
        else:
            raise ValueError("Has not enter into any bucket yet, please call .enter_bucket(bucket_name) to enter a specific bucket")

    def delete_bucket(self, bucket_name: str, interactive: bool = True) -> int:
        """delete the bucket

        :param bucket_name: name of the bucket to delete
        :param interactive: whether you would like an interactive warning message
        :return:
        """
        logging.warning(f"You are about to delete bucket: {bucket_name} and everything under it, please confirm")
        if interactive:
            y_n = input(f"delete everything? (y/n)")
            if y_n in ['y', 'Y']:
                self.resource.Bucket(bucket_name).objects.all().delete()
                self.resource.Bucket(bucket_name).delete()
                return 1
            else:
                logging.warning(f"Aborted by user, please try again")
                return 0

        self.resource.Bucket(bucket_name).delete()
        return 1


    ''' below requires lock a bucket '''

    def list_objs(self, remote_path: str, bucket_name: str = None) -> pd.DataFrame:
        """list objects under the specific directory

        :param remote_path: remote path excluding bucket
        :param bucket_name: name of the bucket, if not given, will use the default bucket
        :return: pandas dataframe showing files inside
        """
        pd.set_option('display.max_colwidth', None)

        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        remote_path = remove_begin_slash(remote_path)

        try:
            res = self.client.list_objects(Bucket = bucket_name, Prefix = remote_path)
        except ClientError as e:
            logging.error(f'Error occurred when listing objects/files under path: {bucket_name}/{remote_path}: \n{e}')
        else:
            if 'Contents' not in res.keys():
                print(f"No objects found at {bucket_name}/{remote_path}")
            else:
                df = pd.DataFrame.from_records(res['Contents'])

                logging.info(f"There are {len(res['Contents'])} files found in s3a://{bucket_name}/{remote_path}")
                return df[['Key', 'LastModified', 'Size']]

    def get_obj(self, remote_path:str, bucket_name: str = None) -> dict:
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        obj = self.client.get_object(Bucket = bucket_name, Key= remove_begin_slash(remote_path))
        return obj

    def create_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Create an object

        :param remote_path: if remote path ends with /, it will be folder, otherwise will be a blank file
        :param bucket_name:
        :return: 1 if success otherwise the info dictionary
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        response  = self.client.put_object(
            Bucket = bucket_name,
            Key = remote_path
        )

        if response.get('ResponseMetadata', {}).get('HTTPStatusCode', {}) == 200:
            logging.info(f"Object Created in Bucket: {bucket_name}, path: {remote_path} ")
            return 1
        else:
            return response

    def delete_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Delete an object/file

        :param remote_path: the filepath to delete (excluding bucket path)
        :param bucket_name: in which bucket
        :return:
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        self.resource.Object(bucket_name, remove_begin_slash(remote_path)).delete()
        return 1

    def create_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Create a folder object on S3

        :param remote_folder_path: the folder path (excluding bucket path)
        :param bucket_name: under which bucket
        :return:
        """
        remote_folder_path = add_tail_slash(remote_folder_path)
        response = self.create_obj(remote_folder_path, bucket_name)
        return response

    def delete_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Delete the folder and every file underneath it

        :param remote_folder_path: the folder path to delete (excluding bucket path)
        :param bucket_name:
        :return:
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        remote_folder_path = add_tail_slash(remote_folder_path)
        self.resource.Bucket(bucket_name).objects.filter(Prefix = remote_folder_path).delete()
        return 1

    def copy_obj(self, from_bucket_name: str = None, from_path: str = None, to_bucket_name: str = None, to_path: str = None):
        """Copy one file from one place to another (can copy to a different bucket)

        :param from_bucket_name: from which bucket you want to copy the file
        :param from_path: which file you want to copy (excluding bucket path)
        :param to_bucket_name: to which bucket you want to copy to
        :param to_path: the destination file path (excluding bucket path)
        :return:
        """
        from_bucket_name = self.if_not_exist_return_default_bucket(from_bucket_name)
        to_bucket_name = self.if_not_exist_return_default_bucket(to_bucket_name)

        copy_source = {
            'Bucket': from_bucket_name,
            'Key': remove_begin_slash(from_path)
        }
        self.client.copy(copy_source, to_bucket_name, remove_begin_slash(to_path))


    def upload_file(self, local_path: str, remote_path: str, bucket_name: str = None, multipart_threshold: int = 1000000000000):
        '''upload file from local machine to S3

        :param local_path: local file path that you would like to upload to S3, suggest to be absolute path
        :param remote_path: remote path on MinIO (excluding bucket path); could either start with / or not
        :param multipart_threshold: Setting an extremely large multipart threshold to effectively disable multipart uplaod.
                Note: Multiplart upload is not compatible with GCS, therefore disabling multipart upload for compatability with hybrid solution.
                set it to None is you want multipart saving
        :return:
        '''
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        try:
            if multipart_threshold:
                config = boto3.s3.transfer.TransferConfig(multipart_threshold = multipart_threshold)
                self.client.upload_file(local_path, bucket_name, key, Config = config)
            else:
                self.client.upload_file(local_path, bucket_name, key)
        except Exception as e:
            logging.error(f'File upload FAILED from {local_path} to {key}:\n'+ str(e))
        else:
            logging.info(f'File uploaded successfully from {local_path} to {key}:')

    def download_file(self, remote_path: str, local_path: str, bucket_name: str = None):
        '''download file from S3 to local machine

        :param remote_path: remote path on S3 (excluding bucket); could either start with / or not
        :param local_path: local file path that you would like to save the file from MinIO, suggest to be absolute path
        :param bucket_name
        :return:
        '''
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        try:
            self.client.download_file(bucket_name, key, local_path)
        except Exception as e:
            logging.error(f'File download FAILED from {key} to {local_path}:\n'+ str(e))
        else:
            logging.info(f'File download successfully from {key} to {local_path}:')


    def read_obj(self, remote_path: str, bucket_name: str = None) -> bytes:
        """read obj from S3 without downloading

        :param remote_path:  remote path to read on S3 (excluding bucket); could either start with / or not
        :param Bucket:
        :return: a buffer (io.StringIO() or io.BytesIO() which can pass to file reader
        """

        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        obj = self.client.get_object(Bucket = bucket_name, Key= remove_begin_slash(remote_path))
        return obj['Body'].read()

    def save_obj(self, body: bytes, remote_path: str, bucket_name: str = None) -> 1:
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        self.resource.Object(bucket_name, remove_begin_slash(remote_path)).put(Body = body)
        return 1

    def read_obj_to_iobuffer(self, remote_path: str, bucket_name: str = None) -> io.BytesIO:
        obj = self.read_obj(remote_path, bucket_name)
        buffer = io.BytesIO(obj)
        return buffer

    def save_iobuffer_to_obj(self, buffer: io.BytesIO, remote_path: str, bucket_name: str = None):
        self.save_obj(buffer.getvalue(), remote_path, bucket_name)

    def read_csv(self, remote_path: str, bucket_name: str = None, *args, **kws) -> pd.DataFrame:
        """Read CSV from S3 directly into Pandas

        :param remote_path: csv file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.read_csv(*args, **kws)
        :param kws: the keyword args passed to pd.read_csv(*args, **kws)
        :return: pandas dataframe
        """
        buffer = self.read_obj_to_iobuffer(remote_path, bucket_name)
        df = pd.read_csv(buffer, *args, **kws)
        return df

    def save_csv(self, df: pd.DataFrame, remote_path: str, bucket_name: str = None, *args, **kws):
        """Save pandas dataframe directly to csv on S3

        :param df: pandas dataframe
        :param remote_path: csv file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.to_csv(*args, **kws)
        :param kws: the keyword args passed to pd.to_csv(*args, **kws)
        :return:
        """
        buffer = io.BytesIO()
        df.to_csv(buffer, *args, **kws)

        self.save_iobuffer_to_obj(buffer, remote_path, bucket_name)

    def read_parquet(self, remote_path: str, bucket_name: str = None, *args, **kws) -> pd.DataFrame:
        """Read Parquet file from S3 directly into Pandas

        :param remote_path: parquet file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.read_parquet(*args, **kws)
        :param kws: the keyword args passed to pd.read_parquet(*args, **kws)
        :return:
        """
        buffer = self.read_obj_to_iobuffer(remote_path, bucket_name)
        df = pd.read_parquet(buffer, *args, **kws)
        return df

    def save_parquet(self, df: pd.DataFrame, remote_path: str, bucket_name: str = None, *args, **kws):
        """Save pandas dataframe directly to parquet file on S3

        :param df: pandas dataframe
        :param remote_path: parquet file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.to_parquet(*args, **kws)
        :param kws: the keyword args passed to pd.to_parquet(*args, **kws)
        :return:
        """
        buffer = io.BytesIO()
        df.to_parquet(buffer, *args, **kws)

        self.save_iobuffer_to_obj(buffer, remote_path, bucket_name)

    def upload_folder(self, local_folder_path: str, remote_root_path: str, bucket_name: str = None, multipart_threshold: int = 1000000000000):
        """Upload a local folder to S3

        :param local_folder_path: local folder path (full path)
        :param remote_root_path: root path where the folder will be created on S3 (excluding bucket path)
        :param bucket_name:
        :param multipart_threshold: an argument controlling uploading size per partition
        :return:
        """
        local_folder_path = os.path.abspath(local_folder_path)
        local_parent_dir_path = os.path.dirname(local_folder_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        for path, subdirs, files in os.walk(local_folder_path):

            dirname = path.replace(local_parent_dir_path, '')
            for file in files:
                local_filepath = os.path.join(path, file)
                remote_filepath = os.path.join(dirname, file)
                if os.path.sep == '\\':
                    remote_filepath = remote_filepath.replace("\\", "/")
                remote_filepath = add_tail_slash(remove_begin_slash(remote_root_path)) +  remove_begin_slash(remote_filepath)

                if multipart_threshold:
                    config = boto3.s3.transfer.TransferConfig(multipart_threshold = multipart_threshold)
                    self.client.upload_file(local_filepath, bucket_name, remote_filepath, Config = config)
                else:
                    self.client.upload_file(local_filepath, bucket_name, remote_filepath)

        logging.info(f"Successfully upload folder from {local_folder_path} to {remote_root_path}")


    def download_folder(self, remote_folder_path: str, local_root_path: str, bucket_name: str = None):
        """

        :param remote_folder_path: folder path on S3 (excluding bucket path)
        :param local_root_path: local root path where the folder will be downloaded to (full root path)
        :param bucket_name:
        :return:
        """
        remote_folder_path = remove_begin_slash(remote_folder_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        for obj in self.resource.Bucket(bucket_name).objects.filter(Prefix = remote_folder_path):
            remote_filepath = obj.key
            local_filepath = os.path.join(local_root_path, os.path.relpath(remote_filepath, remote_folder_path))
            if not os.path.exists(os.path.dirname(local_filepath)):
                os.makedirs(os.path.dirname(local_filepath))
            if remote_filepath.endswith("/"):  # bypass void path
                continue

            self.client.download_file(bucket_name, remote_filepath, local_filepath)

        logging.info(f"Successfully download folder from {remote_folder_path} to {local_root_path}")
