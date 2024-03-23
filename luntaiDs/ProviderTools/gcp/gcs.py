import os
import io
import glob
from typing import Union, Optional
import logging
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

from luntaiDs.CommonTools.obj_storage import ObjStorage, remove_begin_slash, add_tail_slash

class GCSAccessor(ObjStorage):
    def __init__(self, sa_config: Optional[dict] = None, project: str = None, region_name: str = None):
        if not sa_config:
            self.client = storage.Client()
        else:
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                self.client = storage.Client()
            else:
                creds = service_account.Credentials.from_service_account_info(sa_config)
                self.client = storage.Client(credentials=creds)
                
        self.project = project
        self.region_name = region_name
        
    def get_uri_protocol(self) -> str:
        return "gs"
        
    def list_buckets(self) -> pd.DataFrame:
        buckets = self.client.list_buckets()
        props = []
        for bucket in buckets:
            props.append({
                'Name' : bucket._properties["name"],
                'CreationDate' : pd.to_datetime(bucket._properties["timeCreated"])
            })
        return pd.DataFrame.from_records(props)
        
    def create_bucket(self, bucket_name: str) -> int:
        """Create a new bucket

        :param bucket_name: name of the bucket
        :return:
        """
        bucket = self.client.create_bucket(
            bucket_or_name = bucket_name,
            project = self.project,
            location = self.region_name
        )
        return 1
    
    def get_bucket(self, bucket_name: str = None):
        return self.client.get_bucket(bucket_name)
    
    def enter_bucket(self, bucket_name: str = None) -> int:
        """move to a given bucket

        :param bucket_name: name of the bucket you want to switch to
        :return:
        """
        # check if exists
        if self.get_bucket(bucket_name):
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
        """get your current bucket object

        :return:
        """
        if hasattr(self, 'BUCKET'):
            return self.get_bucket(self.BUCKET)
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
                self.get_bucket(bucket_name).delete(force=True)
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
            r = []
            for blob in self.client.list_blobs(bucket_or_name=bucket_name, prefix = remote_path):
                r.append({
                    'Key': blob.name,
                    'LastModified' : blob.updated,
                    'Size': blob.size
                })
        except Exception as e:
            logging.error(f'Error occurred when listing objects/files under path: {bucket_name}/{remote_path}: \n{e}')
        else:
            if len(r) == 0:
                print(f"No objects found at {bucket_name}/{remote_path}")
            else:
                df = pd.DataFrame.from_records(r)

                logging.info(f"There are {len(r)} files found in gcs://{bucket_name}/{remote_path}")
                return df[['Key', 'LastModified', 'Size']]
            
    def get_obj(self, remote_path:str, bucket_name: str = None) -> storage.blob.Blob:
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        obj = bucket.get_blob(blob_name = remove_begin_slash(remote_path))
        return obj
    
    def create_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Create an object

        :param remote_path: if remote path ends with /, it will be folder, otherwise will be a blank file
        :param bucket_name:
        :return: 1 if success otherwise the info dictionary
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(remove_begin_slash(remote_path))
        blob.upload_from_string('', content_type=None)
        return 1
        
    
    def delete_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Delete an object/file

        :param remote_path: the filepath to delete (excluding bucket path)
        :param bucket_name: in which bucket
        :return:
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        obj = bucket.get_blob(blob_name = remove_begin_slash(remote_path))
        obj.delete()
        return 1
    
    def create_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Create a folder object on GCS

        :param remote_folder_path: the folder path (excluding bucket path)
        :param bucket_name: under which bucket
        :return:
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        remote_folder_path = add_tail_slash(remove_begin_slash(remote_folder_path))
        response = self.create_obj(remote_folder_path, bucket_name)
        return response
    
    def delete_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Delete the folder and every file underneath it

        :param remote_folder_path: the folder path to delete (excluding bucket path)
        :param bucket_name:
        :return:
        """
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        remote_folder_path = add_tail_slash(remove_begin_slash(remote_folder_path))
        for blob in bucket.list_blobs(prefix=remote_folder_path):
            blob.delete()
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
        
        source_bucket = self.client.get_bucket(from_bucket_name)
        dest_bucket = self.client.get_bucket(to_bucket_name)
        
        source_blob = source_bucket.blob(remove_begin_slash(from_path))
        source_bucket.copy_blob(
            blob = source_blob, 
            destination_bucket = dest_bucket, 
            new_name = remove_begin_slash(to_path), 
            if_generation_match = 0,
        )
        
    def upload_file(self, local_path: str, remote_path: str, bucket_name: str = None, multipart_threshold: int = 1000000000000):
        '''upload file from local machine to GCS

        :param local_path: local file path that you would like to upload to GCS, suggest to be absolute path
        :param remote_path: remote path on MinIO (excluding bucket path); could either start with / or not
        :param multipart_threshold: Setting an extremely large multipart threshold to effectively disable multipart uplaod.
                Note: Multiplart upload is not compatible with GCS, therefore disabling multipart upload for compatability with hybrid solution.
                set it to None is you want multipart saving
        :return:
        '''
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        try:
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(key)
            blob.upload_from_filename(local_path)
        except Exception as e:
            logging.error(f'File upload FAILED from {local_path} to {key}:\n'+ str(e))
        else:
            logging.info(f'File uploaded successfully from {local_path} to {key}:')
            
    def download_file(self, remote_path: str, local_path: str, bucket_name: str = None):
        '''download file from GCS to local machine

        :param remote_path: remote path on GCS (excluding bucket); could either start with / or not
        :param local_path: local file path that you would like to save the file from MinIO, suggest to be absolute path
        :param bucket_name
        :return:
        '''
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)

        try:
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(key)
            blob.download_to_filename(local_path)
        except Exception as e:
            logging.error(f'File download FAILED from {key} to {local_path}:\n'+ str(e))
        else:
            logging.info(f'File download successfully from {key} to {local_path}:')
            
    def read_obj(self, remote_path: str, bucket_name: str = None) -> bytes:
        """read obj from GCS without downloading

        :param remote_path:  remote path to read on GCS (excluding bucket); could either start with / or not
        :param Bucket:
        :return: a buffer (io.StringIO() or io.BytesIO() which can pass to file reader
        """
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(key)
        return blob.download_as_bytes()
    
    def save_obj(self, body: bytes, remote_path: str, bucket_name: str = None) -> 1:
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(key)
        buffer = io.BytesIO(body)
        blob.upload_from_string(data = buffer.getvalue(), content_type=None)
        return 1
    
    def save_iobuffer_to_obj(self, buffer: io.BytesIO, remote_path: str, bucket_name: str = None):
        key = remove_begin_slash(remote_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(key)
        blob.upload_from_string(data = buffer.getvalue(), content_type=None)
        
    def read_obj_to_iobuffer(self, remote_path: str, bucket_name: str = None) -> io.BytesIO:
        obj = self.read_obj(remote_path, bucket_name)
        buffer = io.BytesIO(obj)
        return buffer
    
    def read_csv(self, remote_path: str, bucket_name: str = None, *args, **kws) -> pd.DataFrame:
        """Read CSV from GCS directly into Pandas

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
        """Save pandas dataframe directly to csv on GCS

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
        """Read Parquet file from GCS directly into Pandas

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
        """Save pandas dataframe directly to parquet file on GCS

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
        """Upload a local folder to GCS

        :param local_folder_path: local folder path (full path)
        :param remote_root_path: root path where the folder will be created on GCS (excluding bucket path)
        :param bucket_name:
        :param multipart_threshold: an argument controlling uploading size per partition
        :return:
        """
        remote_root_path = remove_begin_slash(remote_root_path)
        bucket_name = self.if_not_exist_return_default_bucket(bucket_name)
        bucket = self.client.get_bucket(bucket_name)
        local_parent_dir_path = os.path.dirname(local_folder_path)

        for path, subdirs, files in os.walk(local_folder_path):
            dirname = path.replace(local_parent_dir_path, '')
            for file in files:
                local_filepath = os.path.join(path, file)
                remote_filepath = os.path.join(dirname, file)
                if os.path.sep == '\\':
                    remote_filepath = remote_filepath.replace("\\", "/")
                remote_filepath = add_tail_slash(remove_begin_slash(remote_root_path)) +  remove_begin_slash(remote_filepath)
                # upload
                blob = bucket.blob(remote_filepath)
                blob.upload_from_filename(local_filepath)
        
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
        bucket = self.client.get_bucket(bucket_name)
        
        blobs = bucket.list_blobs(prefix=remote_folder_path)
        for blob in blobs:
            remote_filepath = blob.name
            local_filepath = os.path.join(local_root_path, os.path.relpath(remote_filepath, remote_folder_path))
            if not os.path.exists(os.path.dirname(local_filepath)):
                os.makedirs(os.path.dirname(local_filepath))
            if remote_filepath.endswith("/"):  # bypass void path
                continue
            
            blob.download_to_filename(local_filepath)
            
        logging.info(f"Successfully download folder from {remote_folder_path} to {local_root_path}")