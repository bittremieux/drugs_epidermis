import numpy as np


class CorrelationThreshold:
    
    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else 1.0
    
    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X, rowvar=False))
        self.mask = ~(np.triu(corr, k=1) > self.threshold).any(axis=1)
        return self
    
    def transform(self, X, y=None):
        return X[:, self.mask]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
    
    def get_support(self, indices=False):
        return self.mask if not indices else np.where(self.mask)[0]

