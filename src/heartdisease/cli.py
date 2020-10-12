import click
import click_pathlib
import logging
from heartdisease.models import models_utils

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    pass


@main.command()
@click.option("--data-path", type=click_pathlib.Path(exists=True))
@click.option("--model-version", type=int)
def train_model(data_path, model_version):
    models_utils.run(data_path, model_version)
    logger.info('Finished with training the final model.')


@main.command()
@click.option("--data-path", type=click_pathlib.Path(exists=True))
@click.option("--model-version", type=float)
@click.option("--bucket-name", type=str)
def write_model_aws(data_path, model_version, bucket_name):
    models_utils.write_to_S3(data_path, model_version, bucket_name)
    logger.info('Finished with training the model and writing to S3.')
