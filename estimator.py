import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from features import FeatureExtractor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

class WinProbEstimator:
    def __init__(self):
        self.lgbm_model = LGBMClassifier(max_depth= 5, num_leaves = 7, min_child_samples= 100, learning_rate= 0.05, n_estimators=150, random_state=42, verbose=-1)
        self.calibrated_model = CalibratedClassifierCV(self.lgbm_model, cv=5, method='sigmoid')
        self.extractor = FeatureExtractor()
        
    def fit(self, x, y):
        self.calibrated_model.fit(x, y)
        return self
    
    def evaluate(self, x_test, y_test):
        pred_lgb_probs = self.calibrated_model.predict_proba(x_test)[:, 1]
        print(f"Brier score loss for LGBM Classifier : {brier_score_loss(y_test, pred_lgb_probs):.5f}")

    def predict_match(self, white_elo: int, black_elo: int, opening_moves: str = "", opening_eco: str = "C00") -> float:
        match_pd = pd.DataFrame({'white_rating': white_elo, 'black_rating': black_elo, 'moves': opening_moves, 'opening_eco': opening_eco}, index=[0])
        X_single = self.extractor.transform(match_pd)
        model = self.calibrated_model.predict_proba(X_single)[0, 1]
        return float(model)


        