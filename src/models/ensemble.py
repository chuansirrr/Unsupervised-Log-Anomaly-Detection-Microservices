
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

class HybridAnomalyEnsemble:
    def __init__(self, gmm_components=2, contamination=0.05, random_state=42):
        self.iforest = IsolationForest(contamination=contamination, random_state=random_state)
        self.gmm = GaussianMixture(n_components=gmm_components, random_state=random_state)
        self.calibrated = False

    def fit(self, recon_errors, features):
        self.iforest.fit(features)
        stacked = np.column_stack([recon_errors.reshape(-1,1), -self.iforest.decision_function(features).reshape(-1,1)])
        self.gmm.fit(stacked)
        self.calibrated = True

    def score(self, recon_errors, features):
        iso = -self.iforest.decision_function(features)
        stacked = np.column_stack([recon_errors.reshape(-1,1), iso.reshape(-1,1)])
        nll = -self.gmm.score_samples(stacked) if self.calibrated else stacked.mean(axis=1)
        return nll
