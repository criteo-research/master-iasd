from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, log_loss

from preprocess import timeit


def sanitize_and_split(df: pd.DataFrame, categorical_features: List[str], integer_features: List[str]):
    for f in categorical_features:
        df[f] = df[f].apply(lambda x: x if isinstance(x, str) else "nan")
    return df[integer_features], df[categorical_features], df["label"]


def bucketize(df, qt):
    return pd.DataFrame(np.floor(qt.n_quantiles * qt.transform(df)), columns=df.columns, index=df.index).fillna(-1)


def feature_hashing(df, hasher):
    return hasher.transform((row._asdict() for row in df.itertuples(index=False)))


@timeit
def fit_and_evaluate_model(train_filename: str, test_filename: str, batch_size: int = 128, max_steps: int = 1000):
    integer_features = [f"int_feat_{i}" for i in range(1, 14)]
    categorical_features = [f"cat_feat_{i}" for i in range(1, 27)]

    col_types = {"label": np.bool}
    col_types.update({f: "float32" for f in integer_features})
    col_types.update({f: "str" for f in categorical_features})

    qt = QuantileTransformer(n_quantiles=20)
    hasher = FeatureHasher(n_features=2 ** 16, input_type="dict")
    classifier = linear_model.SGDClassifier(loss="log")

    df_iter_train = pd.read_csv(
        train_filename, sep="\t", header=None, names=col_types.keys(), dtype=col_types, chunksize=batch_size
    )

    for i, df in enumerate(df_iter_train):
        if i % 100 == 0:
            print(f"Fitting on batch {i}")
        integer_features_df, categorical_features_df, label_df = sanitize_and_split(
            df, categorical_features, integer_features
        )

        if i == 0:
            qt.fit(integer_features_df)

        integer_features_df_bucketized = bucketize(integer_features_df, qt)
        df_bucketized = pd.concat([categorical_features_df, integer_features_df_bucketized], axis=1)
        X = feature_hashing(df_bucketized, hasher)
        classifier.partial_fit(X, label_df, classes=[0, 1])

        if i + 1 >= max_steps:
            break

    df_iter_test = pd.read_csv(
        test_filename, sep="\t", header=None, names=col_types.keys(), dtype=col_types, chunksize=batch_size
    )

    auc_scores = []
    log_losses = []

    for i, df in enumerate(df_iter_test):
        if i % 100 == 0:
            print(f"Testing on batch {i}")
        integer_features_df, categorical_features_df, label_df = sanitize_and_split(
            df, categorical_features, integer_features
        )

        integer_features_df_bucketized = bucketize(integer_features_df, qt)
        df_bucketized = pd.concat([categorical_features_df, integer_features_df_bucketized], axis=1)
        X = feature_hashing(df_bucketized, hasher)

        y_pred = classifier.predict_proba(X)[:, 1]

        auc_scores.append(roc_auc_score(label_df, y_pred))
        log_losses.append(log_loss(label_df, y_pred))

    print(f"AUC = {np.mean(auc_scores)}")
    print(f"LogLoss = {np.mean(log_losses)}")


if __name__ == "__main__":
    fit_and_evaluate_model("criteo_train.txt", "criteo_test.txt")
