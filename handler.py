import json
import boto3
import joblib
import pandas as pd
import numpy as np
from io import BytesIO


def get_latest_obj(input_bucket):
    """
    This function gets the last modified file from an S3 bucket.
    :param input_bucket: S3 bucket
    :return: key of the last modified file from the S3 bucket
    """
    get_last_modified = lambda obj: int(obj.last_modified.strftime('%s'))
    objs = [obj for obj in sorted(list(input_bucket.objects.all()), key=get_last_modified)]
    return objs[-1].key

# Connect to S3 bucket with the higher-level object-oriented service access
s3 = boto3.resource('s3')

# Load the specific bucket where the ML model is saved
bucket = s3.Bucket('ml-models-niels')

# Download an write to an in-memory BytesIO object, which acts like a file but doesn't actually touch a disk.
with BytesIO() as file:
    s3.Bucket("ml-models-niels").download_fileobj(get_latest_obj(bucket), file)
    file.seek(0)
    model = joblib.load(file)


def predict(event, context):
    body = {
        "message": "Ok"
    }

    params = event['queryStringParameters']

    x_input = {
        'age': int(params['age']),
        'sex': int(params['sex']),
        'cp': int(params['cp']),
        'trestbps': int(params['trestbps']),
        'chol': int(params['chol']),
        'fbs': int(params['fbs']),
        'restecg': int(params['restecg']),
        'thalach': int(params['thalach']),
        'exang': int(params['exang']),
        'oldpeak': int(np.log1p(params['oldpeak'])),
        'slope': int(params['slope']),
        'ca': int(params['ca']),
        'thal': int(params['thal'])
    }

    X = pd.DataFrame(x_input, index=[0])
    prediction = int(model.predict(X)[0])

    body['predicted heart disease'] = prediction

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }
    return response