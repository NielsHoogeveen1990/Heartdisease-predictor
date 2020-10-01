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
