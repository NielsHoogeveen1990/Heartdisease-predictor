from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import tempfile
import boto3


from heartdisease.models.ML_models import RF
from heartdisease.preprocessing import get_df, get_S3_df


def split_data(df):
    X = df.drop(columns='target')
    y = df['target']

    return train_test_split(X, y, random_state=42)


def fit(model, X_train, y_train):
    clf = model.pipeline()
    clf.fit(X_train, y_train)

    return clf


def evaluate(y_hat, y_true):
    print('accuracy', accuracy_score(y_true, y_hat))
    print('f1_score', f1_score(y_true, y_hat, average='micro'))


def run(datapath, model_version):
    df = get_df(datapath)
    X_train, X_test, y_train, y_test = split_data(df)
    fitted_model = fit(RF, X_train, y_train)
    y_hat = fitted_model.predict(X_test)
    evaluate(y_hat, y_test)


def retrain(datapath, model_version):
    """Train the model on the entire dataset and save it with joblib to a local directory."""
    df = get_df(datapath)
    X = df.drop(columns='target')
    y = df['target']
    fitted_model = fit(RF, X, y)

    with open(f'trained_models/model_{model_version}.pkl', 'wb') as file:
        joblib.dump(fitted_model, file)


def write_to_S3(data_bucket, data_key, model_version, bucket_name):
    """
    Train the model on the entire dataset and sgit bave it in memory to
    subsequently write it so an S3 bucket on AWS.
    """
    df = get_S3_df(data_bucket, data_key)
    X = df.drop(columns='target')
    y = df['target']
    fitted_model = fit(RF, X, y)

    key = f'model_{model_version}.pkl'

    with tempfile.TemporaryFile() as file:
        joblib.dump(fitted_model, file)
        file.seek(0)

        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket_name, key).put(Body=file.read())





