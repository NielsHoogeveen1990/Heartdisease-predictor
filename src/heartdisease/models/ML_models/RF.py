from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

from heartdisease.transformers.dtype_selector import DTypeSelector
from heartdisease.transformers.correlation_filter import CorrFilterHighTotalCorrelation


def pipeline():

    return make_pipeline(
        DTypeSelector('number'),
        CorrFilterHighTotalCorrelation(),
        KNNImputer(n_neighbors=5),
        RobustScaler(),
        RandomForestClassifier()
    )

