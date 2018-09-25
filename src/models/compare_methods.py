# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
from src.features import build_features
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, ShuffleSplit

# TODO: Handle UndefinedMetricWarning
def compare_decision_tree(X, y, n_splits, test_size, metrics):
    """ Runs through k-fold cross validation for the random decision tree
        using the provided data.

        :param X:           np.array of features to train the model
        :param y:           Target variable to predict
        :param n_splits:    Number of splits for cross validation
        :param test_size:   Test size to be passed to ShuffleSplit
        :param metrics:     Tuple of metric names to be passed to cross_val_score

        :return:            summary accuracy metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Compare Decision Tree: Defining ShuffleSplit")
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=n_splits, random_state=0)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    results = None

    for m in metrics:
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring=m)
        except ValueError:
            logger.warning("Compare Decision Tree: %s cannot be calculated via cross_val_score" % m)
            continue
        logger.info("Compare Decision Tree: %s --  %0.4f (+/- %0.4f)" % (m, scores.mean(), scores.std() * 2))

        temp = pd.DataFrame({'model': "Decision Tree", "metric": m, "score": scores})
        if results is None:
            results = pd.DataFrame(temp)
        else:
            results = results.append(temp)

    return results


def compare_random_forest(X, y, n_splits, test_size, metrics):
    """ Runs through k-fold cross validation for the random forest classifier
        using the provided data.

        :param X:           np.array of features to train the model
        :param y:           Target variable to predict
        :param n_splits:    Number of splits for cross validation
        :param test_size:   Test size to be passed to ShuffleSplit
        :param metrics:     Tuple of metric names to be passed to cross_val_score

        :return:            summary accuracy metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Compare Random Forest: Defining ShuffleSplit")
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=n_splits, random_state=0)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    results = None

    for m in metrics:
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring=m)
        except ValueError:
            logger.warning("Compare Random Forest: %s cannot be calculated via cross_val_score" % m)
            continue
        logger.info("Compare Random Forest: %s --  %0.4f (+/- %0.4f)" % (m, scores.mean(), scores.std() * 2))

        temp = pd.DataFrame({'model': "Random Forest", "metric": m, "score": scores})
        if results is None:
            results = pd.DataFrame(temp)
        else:
            results = results.append(temp)

    return results


def compare_logistic_regression(X, y, n_splits, test_size, metrics):
    """ Runs through k-fold cross validation for the logistic regression
        using the provided data.

        :param X:           np.array of features to train the model
        :param y:           Target variable to predict
        :param n_splits:    Number of splits for cross validation
        :param test_size:   Test size to be passed to ShuffleSplit
        :param metrics:     Tuple of metric names to be passed to cross_val_score

        :return:            summary accuracy metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Compare Logistic Regr.: Defining ShuffleSplit")
    logreg = LogisticRegression(C=1e5)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    results = None

    for m in metrics:
        try:
            scores = cross_val_score(logreg, X, y, cv=cv, scoring=m)
        except ValueError:
            logger.warning("Compare Logistic Regr.: %s cannot be calculated via cross_val_score" % m)
            continue
        logger.info("Compare Logistic Regr.: %s --  %0.4f (+/- %0.4f)" % (m, scores.mean(), scores.std() * 2))

        temp = pd.DataFrame({'model': "Logistic Regression", "metric": m, "score": scores})
        if results is None:
            results = pd.DataFrame(temp)
        else:
            results = results.append(temp)

    return results


def main():
    """ Uses the train sample data to run through cross validation of 
        several classification models.
    """

    logger = logging.getLogger(__name__)
    logger.info("Compare Methods: Loading data set")

    n_splits=100
    test_size=0.3
    metrics = ('precision', 'recall', 'f1')

    features = ['ip', 'app', 'device', 'os', 'channel', 'time_x', 'time_y']

    with open("data/processed/train_sample.csv", "r") as f:
        df = pd.read_csv(f)
        f.close()
    
    logger.info("Compare Methods: Adding features.")
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['time_coords'] = df.click_time.apply(build_features.convert_time)
    df['time_x'] = df.time_coords.apply(lambda x: x[0])
    df['time_y'] = df.time_coords.apply(lambda x: x[1])

    df = df.drop(columns='time_coords')

    assert np.isin(features, df.columns).all()
    assert 'is_attributed' in df.columns

    X = df[features].values
    y = df['is_attributed'].values

    results = compare_logistic_regression(X, y, n_splits, test_size, metrics)

    temp = compare_decision_tree(X, y, n_splits, test_size, metrics)
    results = results.append(temp)

    temp = compare_random_forest(X, y, n_splits, test_size, metrics)
    results = results.append(temp)

    logger.info("Compare Methods: Saving results to CSV")
    results.to_csv("models/sample_scores.csv", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()