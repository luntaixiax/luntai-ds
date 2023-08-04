import json
import logging
import os
from collections import OrderedDict



''' ---- 			file loader 			---- '''

def loadJSON(file):
    if isinstance(file, (list, dict)):
        return file
    elif isinstance(file, str):
        with open(file, "rb") as obj:
            return json.load(obj, object_pairs_hook = OrderedDict)

    else:
        raise ValueError("Please parse a file path or JS object")


def toJSON(js, file):
    with open(file, "w") as obj:
        json.dump(js, obj, indent = 4)
