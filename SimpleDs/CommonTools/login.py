from configparser import ConfigParser
from getpass import getpass
from typing import List

from CommonTools.settings import SETTINGS

def get_credential(section: str, attr_keys: List[str] = None):
    """get credentials from credential file specified in Settings

    :param section: which section in the .ini file will extract the credentials
    :param attr_keys: a list of keys, by default will be [username, password]
    :return:
    """
    if not attr_keys:
        attr_keys = ['username', 'password']

    if SETTINGS.DB_CREDENTIAL_INTERACTIVE:
        values = []
        for attr in attr_keys:
            v = getpass(prompt = attr)
            values.append(v)
        return values
    else:
        config = ConfigParser()
        config.read(SETTINGS.DB_CREDENTIAL_PATH)
        sec = config[section]
        values = []
        for attr in attr_keys:
            v = sec[attr]
            values.append(v)
        return values

if __name__ == '__main__':
    SETTINGS.SET_CREDENTIAL_INTERACTIVE()
