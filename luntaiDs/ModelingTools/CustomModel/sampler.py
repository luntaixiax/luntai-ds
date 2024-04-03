import ibis
from ibis import _
from typing import Tuple


class GroupStratifiedSplitterIbis:
    def __init__(self, split_key: str, group_key: str, sample_frac: float = 0.25, train_size: float = 0.8):
        """similar to sklearn group stratified kfold splitter, but apply to Ibis dataset
        the resulting train_ds and test_ds will have following attributes:
        - stratify: equal proportion split_key across train/test set
        - grouping: same split_key entry can either be in train or test set, but not both
        - sampling: total selected train+test size will be sample_frac of original data size
        
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        :param str split_key: the column name for train/test split
        :param str group_key: the column name for group
        :param float sample_frac: down sampling size, defaults to 0.25
        :param float train_size: train size, as % of train + test, defaults to 0.8
        """
        self.split_key = split_key
        self.group_key = group_key
        self.sample_frac = sample_frac
        self.train_size = train_size
        
    def split(self, dataset: ibis.expr.types.Table) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """split into train and test set

        :param ibis.expr.types.Table dataset: the ibis dataset
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: [train_ds, test_ds]
        """
        # assume group_key = CUST_ID, split_key = TARGET, sample_frac = 0.25, train_size = 0.8
        # step 1: split into train/test group, ensure same group col only exist in either train or test
        groups = (
            # step 1.1 group by CUST_ID so that each CUST_ID will only appear in 1 set
            # select CUST_ID, count() as NUM_ from dataset group by CUST_ID
            dataset
            .group_by(self.group_key)
            .aggregate(
                dataset[self.split_key].count().name('NUM_')
            )
            # step 1.2 add a random column for ordering and randomness
            # select cast(hash(CUST_ID) as string) as RAND_ from _
            .mutate(
                _[self.group_key]
                .hash()
                .try_cast("string")
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
        # step 2: stratified sampling
        stratify = (
            dataset
            .left_join(
                groups.select(self.group_key, 'IS_TRAIN_SET_'),
                [self.group_key]
            )
            .drop(f'{self.group_key}_right')
            .mutate(
                (
                    (
                    ibis
                    .row_number()
                    .over(
                        ibis.window(
                            group_by = ['IS_TRAIN_SET_', self.split_key]
                        )
                    ) # partition by target column -- stratify
                    ) / (
                        _
                        .count()
                        .over(
                            ibis.window(
                                group_by = ['IS_TRAIN_SET_', self.split_key]
                            )
                        )
                    ) > self.sample_frac
                ).ifelse(0, 1)
                .name('IS_SELECTED_')
            )
        )
        # step 3: split into train/test set
        train_ds = (
            stratify
            .filter((stratify['IS_SELECTED_'] == 1) & (stratify['IS_TRAIN_SET_'] == 1))
            .drop('IS_SELECTED_', 'IS_TRAIN_SET_')
        )
        test_ds = (
            stratify
            .filter((stratify['IS_SELECTED_'] == 1) & (stratify['IS_TRAIN_SET_'] == 0))
            .drop('IS_SELECTED_', 'IS_TRAIN_SET_')
        )
        # step 4: validate that same CUST_ID only in either train or test but not both
        assert (
            train_ds[train_ds[self.group_key].isin(test_ds[self.group_key])]
            .count()
        ).to_pandas() == 0
        return train_ds, test_ds
        
        