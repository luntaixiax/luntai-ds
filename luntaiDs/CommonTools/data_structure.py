import json
import logging
from collections import OrderedDict
import os


class EnhancedDict(OrderedDict):
    # customized dictionary

    def __init__(self, **kws):
        super().__init__(**kws)

    @ classmethod
    def fromDictSafe(cls, df):
        # convert from python original dictionary to EnhancedDict
        if isinstance(df, EnhancedDict):
            return df

        r = EnhancedDict()
        for k, v in df.items():
            if isinstance(v, dict):
                r[k] = EnhancedDict.fromDictSafe(v)
            else:
                r[k] = v
        return r

    def subset(self, keys):
        # extract specified key-value pairs
        r = EnhancedDict()
        for k, v in self.items():
            if k in keys:
                r[k] = v
        return r

    # override update method
    def update(self, E = None, append = True):
        if append == True:
            super().update(E)
        else:
            # if append = False, only update keys that are in self
            # for keys that are not in self, donnot add them
            for k, v in E.items():
                if k in self:
                    self[k] = v

    def merge(self, E = None, override = True):
        if override == True:
            super().update(E)
        else:
            # if override = False, only add keys that are not in self
            # for keys that are in self, keep the value unchanged
            for k, v in E.items():
                if k not in self:
                    self[k] = v

    def getKeyByValue(self, value, default=""):
        for k, v in self.items():
            if v == value:
                return k
        return default

    def checkValueType(self, vType):
        for k, v in self.items():
            if not isinstance(v, vType):
                raise TypeError("Please Ensure the format for %s is %s descendants!" % (k, vType))

    def applyFunc(self, keyFunc = lambda x:x, valueFunc = lambda x:x, inplace = False):
        '''
        apply uni-variable functions to key and value of the Enhanced dictionary repectively
        @param keyFunc: the univariate function to apply on the key (key transformation)
        @param valueFunc: the univariate function to apply on the value (value transformation)
        @param inplace: when inplace = True, no return and apply on itself, otherwise will return a transformed dict
        @return: only return when inplace = False
        '''
        r = EnhancedDict()
        for k, v in self.items():
            try:
                converted_k = keyFunc(k)
            except:
                logging.critical("Error when applying KeyFunc: %s on key %s; Ignore instead" % (keyFunc, k))
                converted_k = k
            try:
                converted_v = valueFunc(v)
            except:
                logging.critical("Error when applying ValueFunc: %s on key %s; Ignore instead" % (valueFunc, k))
                converted_v = v

            r[converted_k] = converted_v

        if inplace:
            self.clear()
            self.__init__(**r)
            return
        return r

    def println(self):
        for k, v in self.items():
            print(k, " => ", v)

class JsonDict(dict):
    # customized dictionary

    def __init__(self, **kws):
        super().__init__(**kws)

    @ classmethod
    def fromDictSafe(cls, df):
        # convert from python original dictionary to EnhancedDict
        if isinstance(df, JsonDict):
            return df

        if isinstance(df, str):
            df = json.loads(df)

        r = JsonDict()
        for k, v in df.items():
            if isinstance(v, dict):
                r[k] = JsonDict.fromDictSafe(v)
            else:
                r[k] = v
        return r

    def getDict(self, key):
        v = self.get(key)
        return self.fromDictSafe(v)


class Mappers(EnhancedDict):
    def __init__(self, **kws):
        super().__init__()
        for k, v in kws.items():
            self.bindMapper(k, v)

    def bindMapper(self, key, mapper):
        if callable(mapper):
            super().__setitem__(key, mapper)
        else:
            raise TypeError("mapper must be a function or a callable object!")

    def getMapper(self, key):
        return self.get(key)

    __setitem__ = bindMapper  # ensure not adding non-callable values

    __getitem__ = getMapper

    def applyMappers(self, df, defaultMapper=lambda x: x):
        '''
        @param df: input dictionary, e.g. input_dict = {factor1 : 100, factor2 : 200, factor3: 300}
        @param defaultMapper:
        @return:
        # will apply the mappers on each key-value pair in the input dictionary data
        Example:
        input_dict = {factor1 : 100, factor2 : 200, factor3: 300}
        self (mappers) = {factor1 : func1, factor2: func2}
        assume defaultMapper = funcD
        then:
        self.applyMappers(input_dict) -> {factor1 : func1(100), factor2: func2(200), factor3: funcD(300)}

        # Note that the number of key-value pairs in the output will be same as the input dictiondary!
        # use for score mapper!!!!!!!!!!!!!
        '''
        r = EnhancedDict()
        for k, v in df.items():
            mapper = self.getMapper(k)
            if not mapper:
                mapper = defaultMapper
            try:
                mapped_v = mapper(v)
            except:
                logging.critical("Error when apply map %s on key %s; Will keep value before mapping!" % (v, k))
                mapped_v = v
            r[k] = mapped_v
        return r

    def walkMappersOnDict(self, input_dict):
        '''
        @param input_dict: input dictionary, e.g. input_dict = {a : 4, b : 5, c : 6, d : 7}
        @return:
        # will apply all the mappers on one single dictionary data
        # assumes each mapper takes the whole dictionary data df as input
        Example:
        input_dict = {a : 4, b : 5, c : 6, d : 7}
        self (mappers) = {factor1 : func1, factor2: func2}
        then:
        self.walkMappersOnDict(input_dict) -> {factor1 : func1(input_dict), factor2: func2(input_dict)}

        # Note that the number of key-value pairs in the output will be same as the mapper itself!
        # use for factor mapper!!!!!!!!!!!!!
        '''
        factors = self.applyFunc(valueFunc=lambda mapper: mapper(input_dict))
        return factors
    

class SingleObjDataManager:
    def __init__(self, root_dir: str):
        self.ROOT_DIR = root_dir

    def get_path(self, *relative_path) -> str:
        return os.path.join(self.ROOT_DIR, *relative_path)

    def is_exist(self, *relative_path) -> bool:
        return os.path.exists(self.get_path(*relative_path))