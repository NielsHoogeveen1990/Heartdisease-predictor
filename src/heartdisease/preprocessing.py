import numpy as np
import pandas as pd
from heartdisease.utils import log_step


def read_data(data_path):
    """
    This function reads the data from a given data path.
    :param data_path: data path to the CSV source file.
    :return: dataframe
    """
    return pd.read_csv(data_path, error_bad_lines=False).drop_duplicates()


def log_transform(df):
    return df.assign(
        oldpeak=lambda d: np.log1p(d['oldpeak'])
    )


@log_step
def get_df(data_path):
    return (read_data(data_path)
            .pipe(log_transform)
            )
