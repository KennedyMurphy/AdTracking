# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
from src.features import build_features
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, ShuffleSplit
from src.features import build_features

MODEL_PATH = 'models/trained_model.pickle'

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

logger = logging.getLogger(__name__)
logger.info("Prediction: Reading in trained model from %s" % MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
    f.close()

# Read in the data
df = build_features.feature_creation("data/processed/test.csv")

# Extract the features for predicting
features = [c for c in df.columns if c != 'click_id']
logger.info("Prediction: Generating predictions with %s" % ", ".join(features))
X = df[features].values

# Remove the original data frame from memory
del df

logger.info("Prediction: Generating predictions")
y_pred = clf.predict(X)
y_pred = pd.DataFrame({'click_id': range(len(y_pred)), 'is_attributed': y_pred})

logger.info("Prediction: Saving predictions to CSV")
y_pred.to_csv('models/predictions.csv', index=False)