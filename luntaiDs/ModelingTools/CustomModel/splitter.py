import ibis
from ibis import _
from typing import Tuple

class SimpleSplitterIbis:
    def __init__(self, shuffle_key: str, train_size: float = 0.8, 
                 random_seed: int = 0, verbose: bool = True):
        """split into train and test set

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param float train_size: train sample size as of total size, defaults to 0.8
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        self.random_seed = random_seed
        self.verbose = verbose
        self.shuffle_key = shuffle_key
        if train_size > 1 or train_size < 0:
            raise ValueError("train_size can only between (0,1)")
        self.train_size = train_size
        
    def split(self, dataset: ibis.expr.types.Table) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """split into train and test set

        :param ibis.expr.types.Table dataset: the ibis dataset
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: [train_ds, test_ds]
        """
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        dataset = (
            dataset
            .order_by(
                (_[self.shuffle_key]
                .try_cast("string") + str(self.random_seed)) # randomness here
                .hash()
            )
            .mutate(
                (ibis.row_number().over() 
                / _.count().over() > self.train_size)
                .ifelse(0, 1)
                .name('IS_TRAIN_SET_')
            )
        )
        # step 3: split into train/test set
        train_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 1)
            .drop('IS_TRAIN_SET_')
        )
        test_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 0)
            .drop('IS_TRAIN_SET_')
        )
        
        return train_ds, test_ds
        

class StratifiedSplitterIbis(SimpleSplitterIbis):
    def __init__(self, shuffle_key: str, stratify_key: str, train_size: float = 0.8, 
                 random_seed: int = 0, verbose: bool = True):
        """split into train and test set, stratified on given stratify_key

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param str stratify_key: the column name for determine stratify
        :param float train_size: train sample size as of total size, defaults to 0.8
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        super().__init__(
            train_size = train_size,
            shuffle_key = shuffle_key,
            random_seed = random_seed,
            verbose = verbose
        )
        self.stratify_key = stratify_key
        
    def split(self, dataset: ibis.expr.types.Table) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """split into train and test set

        :param ibis.expr.types.Table dataset: the ibis dataset
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: [train_ds, test_ds]
        """
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        if self.stratify_key not in dataset.columns:
            raise KeyError(f"Stratify key {self.stratify_key} not found in dataset")
        dataset = (
            dataset
            .mutate(
                (_[self.shuffle_key]
                .try_cast("string") + str(self.random_seed)) # randomness here
                .hash()
                .name('RAND_')
            )
            .mutate(
                (
                    (
                    ibis
                    .row_number()
                    .over(
                        ibis.window(
                            group_by = [self.stratify_key],
                            order_by = ['RAND_']
                        )
                    ) # partition by target column -- stratify
                    ) / (
                        _
                        .count()
                        .over(
                            ibis.window(
                                group_by = [self.stratify_key]
                            )
                        )
                    ) > self.train_size
                ).ifelse(0, 1)
                .name('IS_TRAIN_SET_')
            )
        )
        # step 3: split into train/test set
        train_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 1)
            .drop('IS_TRAIN_SET_', 'RAND_')
        )
        test_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 0)
            .drop('IS_TRAIN_SET_', 'RAND_')
        )
        
        return train_ds, test_ds

class GroupSplitterIbis(SimpleSplitterIbis):
    def __init__(self, shuffle_key: str, group_key: str, train_size: float = 0.8,
                 random_seed: int = 0, verbose: bool = True):
        """split into train and test set, each group_key will only appear in either train or test set, not both

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param str group_key: the column name for determine group of each sample
        :param float train_size: train sample size as of total size, defaults to 0.8
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        super().__init__(
            train_size = train_size,
            shuffle_key = shuffle_key,
            random_seed = random_seed,
            verbose = verbose
        )
        self.group_key = group_key
        
    def split(self, dataset: ibis.expr.types.Table) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """split into train and test set

        :param ibis.expr.types.Table dataset: the ibis dataset
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: [train_ds, test_ds]
        """
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        if self.group_key not in dataset.columns:
            raise KeyError(f"Group key {self.group_key} not found in dataset")
        # assume group_key = CUST_ID, train_size = 0.8
        # step 1: split into train/test group, ensure same group col only exist in either train or test
        groups = (
            # step 1.1 group by CUST_ID so that each CUST_ID will only appear in 1 set
            # select CUST_ID, count() as NUM_ from dataset group by CUST_ID
            dataset
            .group_by(self.group_key)
            .aggregate(
                _.count().name('NUM_')
            )
            # step 1.2 add a random column for ordering and randomness
            # select cast(hash(CUST_ID) as string) as RAND_ from _
            .mutate(
                (_[self.shuffle_key]
                .try_cast("string") + str(self.random_seed)) # randomness here
                .hash()
                .name('RAND_')
            )
            # step 1.3 add a divider between train and test set
            # select CUST_ID, if(sum(NUM_) over (order by RAND_) / sum() over() > 0.25, 0, 1) as IS_TRAIN_SET_
            .mutate(
                (
                    (
                    _['NUM_']
                    .sum()
                    .over(
                        ibis.window(
                            order_by = 'RAND_',
                            following = 0
                        )
                    ) # partition by target column -- stratify
                    ) / (
                        _['NUM_']
                        .sum()
                        .over()
                    ) > self.train_size
                ).ifelse(0, 1)
                .name('IS_TRAIN_SET_')
            )
        )
        groups = (
            dataset
            .left_join(
                groups.select(self.group_key, 'IS_TRAIN_SET_'),
                [self.group_key]
            )
            .drop(f'{self.group_key}_right')
        )
        # step 3: split into train/test set
        train_ds = (
            groups
            .filter(groups['IS_TRAIN_SET_'] == 1)
            .drop('IS_TRAIN_SET_')
        )
        test_ds = (
            groups
            .filter(groups['IS_TRAIN_SET_'] == 0)
            .drop('IS_TRAIN_SET_')
        )
        # step 4: validate that same CUST_ID only in either train or test but not both
        assert (
            train_ds[train_ds[self.group_key].isin(test_ds[self.group_key])]
            .count()
        ).to_pandas() == 0, f"Found some {self.group_key} in both train and test set"
        
        return train_ds, test_ds
    
    
class TimeSeriesSplitterIbis(SimpleSplitterIbis):
    def __init__(self, shuffle_key: str, ts_key: str, train_size: float = 0.8, 
                 random_seed: int = 0, verbose: bool = True):
        """split into train and test set, but test will be later than train on ts_key

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param str ts_key: time series column indicating the order of dataframe
        :param float train_size: train sample size as of total size, defaults to 0.8
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        super().__init__(
            train_size = train_size,
            shuffle_key = shuffle_key,
            random_seed = random_seed,
            verbose = verbose
        )
        self.ts_key = ts_key
        
    def split(self, dataset: ibis.expr.types.Table) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """split into train and test set

        :param ibis.expr.types.Table dataset: the ibis dataset
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: [train_ds, test_ds]
        """
        if self.ts_key not in dataset.columns:
            raise KeyError(f"Time Series key {self.ts_key} not found in dataset")
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        dataset = (
            dataset
            .mutate(
                (_[self.shuffle_key]
                .try_cast("string") + str(self.random_seed)) # randomness here
                .hash()
                .name('RAND_')
            )
            .order_by([self.ts_key, 'RAND_'])
            .mutate(
                (ibis.row_number().over() 
                / _.count().over() > self.train_size)
                .ifelse(0, 1)
                .name('IS_TRAIN_SET_')
            )
        )
        train_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 1)
            .drop('IS_TRAIN_SET_', 'RAND_')
        )
        test_ds = (
            dataset
            .filter(dataset['IS_TRAIN_SET_'] == 0)
            .drop('IS_TRAIN_SET_', 'RAND_')
        )
        assert (
            train_ds[self.ts_key].max().to_pandas() <= test_ds[self.ts_key].min().to_pandas()
        ), f"There are some records in train set that have {self.ts_key} later than that in test set"
        
        return train_ds, test_ds