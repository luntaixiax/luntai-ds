import io
from typing import List, Any
import pandas as pd

def remove_begin_slash(url: str) -> str:
    if url.startswith('/'):
        prefix = str(url)[1:]
    else:
        prefix = str(url)
    return prefix

def add_tail_slash(url: str) -> str:
    if url.endswith('/'):
        return str(url)
    else:
        return str(url) + '/'

class ObjStorage:
    
    def get_uri_protocol(self) -> str:
        raise NotImplementedError("")

    def list_buckets(self) -> pd.DataFrame:
        """return bucket list [Name, CreationDate]
        """
        raise NotImplementedError("")

    def create_bucket(self, bucket_name: str) -> int:
        """Create a new bucket

        :param bucket_name: name of the bucket
        :return: 1 if success
        """
        raise NotImplementedError("")

    def get_bucket(self, bucket_name: str = None):
        raise NotImplementedError("")


    def enter_bucket(self, bucket_name: str = None) -> int:
        """move to a given bucket

        :param bucket_name: name of the bucket you want to switch to
        :return: 1 if success
        """
        raise NotImplementedError("")

    def get_current_bucket(self):
        """get your current bucket object

        :return:
        """
        raise NotImplementedError("")

    def delete_bucket(self, bucket_name: str, interactive: bool = True) -> int:
        """delete the bucket

        :param bucket_name: name of the bucket to delete
        :param interactive: whether you would like an interactive warning message
        :return: 1 if success
        """
        raise NotImplementedError("")


    ''' below requires lock a bucket '''

    def list_objs(self, remote_path: str, bucket_name: str = None) -> pd.DataFrame:
        """list objects under the specific directory

        :param remote_path: remote path excluding bucket
        :param bucket_name: name of the bucket, if not given, will use the default bucket
        :return: pandas dataframe showing files inside ['Key', 'LastModified', 'Size']
        """
        raise NotImplementedError("")

    def get_obj(self, remote_path:str, bucket_name: str = None) -> Any:
        raise NotImplementedError("")

    def create_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Create an object

        :param remote_path: if remote path ends with /, it will be folder, otherwise will be a blank file
        :param bucket_name:
        :return: 1 if success otherwise the info dictionary
        """
        raise NotImplementedError("")

    def delete_obj(self, remote_path: str, bucket_name: str = None) -> int:
        """Delete an object/file

        :param remote_path: the filepath to delete (excluding bucket path)
        :param bucket_name: in which bucket
        :return:
        """
        raise NotImplementedError("")

    def create_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Create a folder object on object storage

        :param remote_folder_path: the folder path (excluding bucket path)
        :param bucket_name: under which bucket
        :return:
        """
        raise NotImplementedError("")

    def delete_folder(self, remote_folder_path: str, bucket_name: str = None) -> int:
        """Delete the folder and every file underneath it

        :param remote_folder_path: the folder path to delete (excluding bucket path)
        :param bucket_name:
        :return:
        """
        raise NotImplementedError("")

    def copy_obj(self, from_bucket_name: str = None, from_path: str = None, to_bucket_name: str = None, to_path: str = None):
        """Copy one file from one place to another (can copy to a different bucket)

        :param from_bucket_name: from which bucket you want to copy the file
        :param from_path: which file you want to copy (excluding bucket path)
        :param to_bucket_name: to which bucket you want to copy to
        :param to_path: the destination file path (excluding bucket path)
        :return:
        """
        raise NotImplementedError("")


    def upload_file(self, local_path: str, remote_path: str, bucket_name: str = None, multipart_threshold: int = 1000000000000):
        '''upload file from local machine to object storage

        :param local_path: local file path that you would like to upload to S3, suggest to be absolute path
        :param remote_path: remote path on MinIO (excluding bucket path); could either start with / or not
        :param multipart_threshold: Setting an extremely large multipart threshold to effectively disable multipart uplaod.
                Note: Multiplart upload is not compatible with GCS, therefore disabling multipart upload for compatability with hybrid solution.
                set it to None is you want multipart saving
        :return:
        '''
        raise NotImplementedError("")

    def download_file(self, remote_path: str, local_path: str, bucket_name: str = None):
        '''download file from object storage to local machine

        :param remote_path: remote path on object storage (excluding bucket); could either start with / or not
        :param local_path: local file path that you would like to save the file from MinIO, suggest to be absolute path
        :param bucket_name
        :return:
        '''
        raise NotImplementedError("")


    def read_obj(self, remote_path: str, bucket_name: str = None) -> bytes:
        """read obj from object storage without downloading

        :param remote_path:  remote path to read on object storage (excluding bucket); could either start with / or not
        :param Bucket:
        :return: a buffer (io.StringIO() or io.BytesIO() which can pass to file reader
        """
        raise NotImplementedError("")

    def save_obj(self, body: bytes, remote_path: str, bucket_name: str = None) -> 1:
        raise NotImplementedError("")

    def read_obj_to_iobuffer(self, remote_path: str, bucket_name: str = None) -> io.BytesIO:
        raise NotImplementedError("")

    def save_iobuffer_to_obj(self, buffer: io.BytesIO, remote_path: str, bucket_name: str = None):
        raise NotImplementedError("")

    def read_csv(self, remote_path: str, bucket_name: str = None, *args, **kws) -> pd.DataFrame:
        """Read CSV from S3 directly into Pandas

        :param remote_path: csv file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.read_csv(*args, **kws)
        :param kws: the keyword args passed to pd.read_csv(*args, **kws)
        :return: pandas dataframe
        """
        raise NotImplementedError("")

    def save_csv(self, df: pd.DataFrame, remote_path: str, bucket_name: str = None, *args, **kws):
        """Save pandas dataframe directly to csv on object storage

        :param df: pandas dataframe
        :param remote_path: csv file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.to_csv(*args, **kws)
        :param kws: the keyword args passed to pd.to_csv(*args, **kws)
        :return:
        """
        raise NotImplementedError("")

    def read_parquet(self, remote_path: str, bucket_name: str = None, *args, **kws) -> pd.DataFrame:
        """Read Parquet file from object storage directly into Pandas

        :param remote_path: parquet file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.read_parquet(*args, **kws)
        :param kws: the keyword args passed to pd.read_parquet(*args, **kws)
        :return:
        """
        raise NotImplementedError("")

    def save_parquet(self, df: pd.DataFrame, remote_path: str, bucket_name: str = None, *args, **kws):
        """Save pandas dataframe directly to parquet file on object storage

        :param df: pandas dataframe
        :param remote_path: parquet file path (excluding bucket path)
        :param bucket_name:
        :param args: the positional args passed to pd.to_parquet(*args, **kws)
        :param kws: the keyword args passed to pd.to_parquet(*args, **kws)
        :return:
        """
        raise NotImplementedError("")

    def upload_folder(self, local_folder_path: str, remote_root_path: str, bucket_name: str = None, multipart_threshold: int = 1000000000000):
        """Upload a local folder to object storage

        :param local_folder_path: local folder path (full path)
        :param remote_root_path: root path where the folder will be created on S3 (excluding bucket path)
        :param bucket_name:
        :param multipart_threshold: an argument controlling uploading size per partition
        :return:
        """
        raise NotImplementedError("")


    def download_folder(self, remote_folder_path: str, local_root_path: str, bucket_name: str = None):
        """

        :param remote_folder_path: folder path on object storage (excluding bucket path)
        :param local_root_path: local root path where the folder will be downloaded to (full root path)
        :param bucket_name:
        :return:
        """
        raise NotImplementedError("")
