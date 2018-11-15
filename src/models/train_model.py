# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.features import build_features
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier

FILE_NAME = 'train_sample.csv'
FEATURES = ['ip', 'app', 'device', 'os', 'channel']
N_SPLITS = 50
N_ESTIMATORS = 10

def main():
    """ Manages feature extraction and model training.
    """
    logger = logging.getLogger(__name__)
    logger.info("Train Model: Loading data from %s" % FILE_NAME)
    df = pd.read_csv("data/processed/%s" % FILE_NAME)

    # Make sure that all FEATURES are in the data frame.
    assert np.isin(FEATURES, df.columns).all()
    assert 'is_attributed' in df.columns

    logger.info("Train Model: Measuring class balance.")
    count_class_0, count_class_1 = df.is_attributed.value_counts()

    df_class_0 = df[df['is_attributed'] == 0]
    df_class_1 = df[df['is_attributed'] == 1]

    df = pd.concat([df_class_1, df_class_0.sample(count_class_1)], axis=0)
    new_class_0, new_class_1 = df.is_attributed.value_counts()

    logger.info("Train Model: Resampled Class 0 (%i -> %i); Class 1 (%i -> %i)" 
                % (count_class_0, new_class_0, count_class_1, new_class_1))

    # Remove the temporary tables from memory
    del df_class_0
    del df_class_1

    logger.info("Train Model: Extracting features %s" % ", ".join(FEATURES))
    X = df[FEATURES].values
    y = df['is_attributed'].values

    # Remove the original data frame from memory
    del df

    logger.info("Train Model: Building RFC with n_estimators (%i) and n_splits (%i)" %
                (N_ESTIMATORS, N_SPLITS))
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=N_SPLITS,
        random_state=0)

    logger.info("Train Model: Beginning training cycle.")
    clf.fit(X, y)

    logger.info("Train Model: Saving model for records.")
    with open("models/trained_model.pickle", "wb") as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        f.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()