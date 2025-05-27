import joblib
import numpy as np

class CombinedModel:
    def __init__(self):
        self.classifier = joblib.load("models/classifier.pkl")  # existing model
        self.regressor = joblib.load("models/next_close_regressor.pkl")

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict_next_close(self, X):
        return self.regressor.predict(X)
