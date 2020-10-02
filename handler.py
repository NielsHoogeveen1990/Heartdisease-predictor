import json
import boto3
import joblib
import pandas as pd
from io import BytesIO
import os

s3 = boto3.resource(
    's3',
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

with BytesIO() as file:
    s3.Bucket("ml-models-niels").download_fileobj("model_1.5.joblib", file)
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
        'oldpeak': int(params['oldpeak']),
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