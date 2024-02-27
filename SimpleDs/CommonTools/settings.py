
class SETTINGS:
    DB_CREDENTIAL_PATH = None
    DB_CREDENTIAL_INTERACTIVE = False

    HTTP_PROXY = None
    HTTPS_PROXY = None

    @ classmethod
    def SET_CREDENTIAL_INTERACTIVE(cls):
        """set to interactive mode, good for JupyterHub

        :return:
        """
        cls.DB_CREDENTIAL_INTERACTIVE = True
        cls.DB_CREDENTIAL_PATH = None

    @ classmethod
    def SET_CREDENTIAL_PATH(cls, path = None):
        """set to script mode, and specify a credential ini file path

        :param path: should be an .ini or .toml configuration file following the below format:
        :return:

        example credential.ini:
        [Database]
        username = xxxxxxx
        password = xxxxxxx
        """
        cls.DB_CREDENTIAL_INTERACTIVE = False
        cls.DB_CREDENTIAL_PATH = path

    @ classmethod
    def SET_PROXY_SERVER(cls, http_proxy: str, https_proxy: str):
        cls.HTTP_PROXY = http_proxy
        cls.HTTPS_PROXY = https_proxy