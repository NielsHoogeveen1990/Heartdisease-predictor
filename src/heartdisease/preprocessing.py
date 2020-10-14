import numpy as np
import pandas as pd
import boto3
from heartdisease.utils import log_step


def read_data(data_path):
    """
    This function reads the data from a given data path.
    :param data_path: data path to the CSV source file.
    :return: dataframe
    """
    return pd.read_csv(data_path, error_bad_lines=False).drop_duplicates()


def read_S3_data(bucket, key):
    """
    This function reads CSV data from an S3 bucket.
    :param bucket: S3 bucket
    :param key: name of the data file in the S3 bucket
    :return: dataframe
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], error_bad_lines=False).drop_duplicates()


def log_transform(df):
    return df.assign(
        oldpeak=lambda d: np.log1p(d['oldpeak'])
    )


@log_step
def get_df(data_path):
    return (read_data(data_path)
            .pipe(log_transform)
            )


@log_step
def get_S3_df(bucket, key):
    return (read_S3_data(bucket, key)
            .pipe(log_transform)
            )
