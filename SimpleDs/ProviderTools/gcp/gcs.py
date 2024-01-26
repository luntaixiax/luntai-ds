import os
from typing import Union, Optional
import logging
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

class GCSAccessor:
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
        
    def list_buckets(self) -> pd.DataFrame:
        buckets = self.client.list_buckets()
        
    def create_bucket(self, bucket_name: str, region_name: str = 'US-CENTRAL1') -> int:
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
                self.resource.Bucket(bucket_name).objects.all().delete()
                self.resource.Bucket(bucket_name).delete()
                return 1
            else:
                logging.warning(f"Aborted by user, please try again")
                return 0

        self.resource.Bucket(bucket_name).delete()
        return 1
    
    ''' below requires lock a bucket '''
    