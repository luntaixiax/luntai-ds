from functools import partial
import pandas as pd

def _select_present_subset(selected_columns, df: pd.DataFrame):
    return [col for col in df.columns if col in selected_columns]

def make_present_col_selector(selected_columns):
    """ColumnTransformer support handling of missing columns

    :param selected_columns:
    :return: a column selector that can be fitted to ColumnTransformer

    Reference:
    https://github.com/scikit-learn/scikit-learn/issues/19014
    https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer

    e.g.
    preprocessor_new = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_present_col_selector(numeric_features)),
        ('cat', categorical_transformer, make_present_col_selector(categorical_features))
    ]
)
    """
    return partial(_select_present_subset, selected_columns)


def _select_present_sibling_subset(selected_columns, df: pd.DataFrame):
    # find the columns from df that are similar (siblings) of selected columns
    # sometimes column changes as result of onehot encoding and imputer indicator
    return df.columns[df.columns.str.contains('|'.join(selected_columns))] # use a regrex to define

def make_sibling_col_selector(selected_columns):
    return partial(_select_present_sibling_subset, selected_columns)