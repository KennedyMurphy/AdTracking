# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
from src.features import build_features
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, ShuffleSplit
from src.models.train_model import FEATURES

MODEL_PATH = 'models/trained_model.pickle'

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

logger = logging.getLogger(__name__)
logger.info("Prediction: Reading in trained model from %s" % MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
    f.close()

logger.info("Prediction: Reading in test set features.")
df = pd.read_csv("data/processed/test.csv")

# Make sure that all FEATURES are in the data frame.
assert np.isin(FEATURES, df.columns).all()

logger.info("Prediction: Extracting features %s" % ", ".join(FEATURES))
X = df[FEATURES].values

# Remove the original data frame from memory
del df

logger.info("Prediction: Generating predictions")
y_pred = clf.predict(X)