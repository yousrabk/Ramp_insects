import os
os.environ["THEANO_FLAGS"] = "device=gpu"

import numpy as np

from sklearn.pipeline import make_pipeline
from caffezoo.googlenet import GoogleNet
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Classifier(BaseEstimator):

    def __init__(self):
        self.clf = make_pipeline(
            GoogleNet(layer_names=["inception_4e/output"]),
            RandomForestClassifier(n_estimators=100, max_depth=25)
        )

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
