import json
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any

class AirflowAPI:
    # https://airflow.apache.org/docs/apache-airflow/stable/security/api.html#api-authentication
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self._base_url = f"http://{host}:{port}/api/v1"
        self._auth = HTTPBasicAuth(username, password)
        
    def get_variable_list(self) -> List[str]:
        r = requests.get(
            url = f"{self._base_url}/variables",
            auth = self._auth
        )
        if r.status_code != 200:
            r.raise_for_status()
        r = r.json().get('variables')
        return [i.get("key") for i in r]
    
    def get_variable(self, key: str) -> Any:
        r = requests.get(
            url = f"{self._base_url}/variables/{key}",
            auth = self._auth
        )
        if r.status_code != 200:
            r.raise_for_status()
        return r.json().get('value')
    
    def upsert_variable(self, key: str, value: Any, descr: str = ""):
        # see if in variable already
        vars = self.get_variable_list()
        if key in vars:
            # update
            r = requests.patch(
                url = f"{self._base_url}/variables/{key}",
                auth = self._auth,
                json = {
                    "description": descr,
                    "key": key,
                    "value": json.dumps(value)
                }
            )
        else:
            # insert
            r = requests.post(
                url = f"{self._base_url}/variables",
                auth = self._auth,
                json = {
                    "description": descr,
                    "key": key,
                    "value": json.dumps(value)
                }
            )
        if r.status_code != 200:
            r.raise_for_status()
            
    def delete_variable(self, key: str):
        r = requests.delete(
            url = f"{self._base_url}/variables/{key}",
            auth = self._auth
        )
        if r.status_code != 200:
            r.raise_for_status()
    