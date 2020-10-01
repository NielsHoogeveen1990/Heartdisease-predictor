from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class DTypeSelector(BaseEstimator, TransformerMixin):
    """
    This transformer selects columns based on their datatype.
    Selecting columns of a certain datatype is useful in a pipeline where one wants to apply
    different transformers (e.g. different imputation techniques) for different datatypes.
    """

    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, X, y=None):
        """
        The fit method of an DTypeSelector object simply returns self and validates whether X is an instance of a dataframe.
        :param X: X dataframe
        :param y: y series
        :return: self
        """
        self._validate_X(X)
        return self

    def transform(self, X):
        """
        This method will select and return only the columns with the specified datatype.
        :param X: X dataframe
        :return: X dataframe with columns of a specific datatype
        """
        return X.select_dtypes(self.dtypes)

    @staticmethod
    def _validate_X(X):
        """
        This static method validates whether X is a Pandas dataframe.
        :param X: X dataframe
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a dataframe.")

