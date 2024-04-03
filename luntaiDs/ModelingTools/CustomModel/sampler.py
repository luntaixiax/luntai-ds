import ibis
from ibis import _

class SimpleSamplerIbis:
    def __init__(self, shuffle_key: str, sample_frac: float = 0.25, 
                 random_seed: int = 0, verbose: bool = True):
        """down sample without any constraint

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param float sample_frac: down sampling size, defaults to 0.25
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        self.random_seed = random_seed
        self.verbose = verbose
        self.shuffle_key = shuffle_key
        if sample_frac > 1 or sample_frac <= 0:
            raise ValueError("sample frac can only between (0,1]")
        self.sample_frac = sample_frac
        
    def sample(self, dataset: ibis.expr.types.Table) -> ibis.expr.types.Table:
        """execute the sampling

        :param ibis.expr.types.Table dataset: ibis dataset
        :return ibis.expr.types.Table: sampled ibis dataset
        """
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        if self.sample_frac == 1:
            return dataset
        
        selected = (
            dataset
            .order_by(
                (_[self.shuffle_key]
                .try_cast("string") + str(self.random_seed)) # randomness here
                .hash()
            )
            .mutate(
                (ibis.row_number().over() 
                / _.count().over() > self.sample_frac)
                .ifelse(0, 1)
                .name('IS_SELECTED_')
            )
            .filter(
                _['IS_SELECTED_'] == 1
            ).drop('IS_SELECTED_')
        )
        
        return selected


class StratifiedSamplerIbis(SimpleSamplerIbis):
    def __init__(self, shuffle_key: str, stratify_key: str, sample_frac: float = 0.25, 
                 random_seed: int = 0, verbose: bool = True):
        """down sample with roughly equal distributon of split_key before and after sampling

        :param str shuffle_key: id column for identitying each row, if given, will shuffle
        :param str stratify_key: the column name for determine stratify
        :param float sample_frac: down sampling size, defaults to 0.25
        :param int random_seed: control randomness
        :param bool verbose: whether to print out validation message and statistics
        """
        super().__init__(
            sample_frac = sample_frac,
            shuffle_key = shuffle_key,
            random_seed = random_seed,
            verbose = verbose
        )
        self.stratify_key = stratify_key

        
    def sample(self, dataset: ibis.expr.types.Table) -> ibis.expr.types.Table:
        """execute the sampling

        :param ibis.expr.types.Table dataset: ibis dataset
        :return ibis.expr.types.Table: sampled ibis dataset
        """
        if self.shuffle_key not in dataset.columns:
            raise KeyError(f"Shuffle key {self.shuffle_key} not found in dataset")
        if self.stratify_key not in dataset.columns:
            raise KeyError(f"Stratify key {self.stratify_key} not found in dataset")
        if self.sample_frac == 1:
            return dataset
        
        if self.shuffle_key:
            dataset = (
                dataset
                
            )
        
        selected = (
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
                    ) > self.sample_frac
                ).ifelse(0, 1)
                .name('IS_SELECTED_')
            )
        )
        return (
            selected
            .filter(selected['IS_SELECTED_'] == 1)
            .drop('IS_SELECTED_', 'RAND_')
        )