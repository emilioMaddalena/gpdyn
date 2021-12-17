#%%

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import json

#import gpflow
#import seaborn as sns
#import tensorflow as tf
#import tensorflow_probability as tfp
#from gpflow.utilities import print_summary, set_trainable
#from sklearn.metrics import pairwise_distances 

class GpDynModel:
    """
    An auto-regressive Gaussian process model with exogenous inputs.
    """

    def __init__(self, name=''):
        
        # external attributes
        self.name = name

        # internal attributes
        self._data_was_loaded = False
        self._data_was_filtered = False

    def load_data(self, X, Y, delays):
        """
        Args:
            X: np.array of shape (S, N). S is the number of signals and N is the number of data points.
            Y: np.array of shape (1, N). N is the number of data points.
        """

        self._check_shapes(X, Y)
        self._check_delays(X, delays)
        delays = self._transform_delays(X, delays)

        self.X = X
        self.Y = Y
        self.delays = delays

        self._data_was_loaded = True

    def train_model(self, kernel, **kwargs):

        if not self._data_was_loaded: 
            print('Call load_data() before calling train_model().')
            return

        self._model_was_trained = True

    def predict(self):
        pass

    def _check_shapes(self, X, Y, delays=0):

        if len(X.shape) != 2: 
            raise ValueError('Incorrect shape for X. Must be (S, N), S = number of signals and N = number of samples.')
        
        if (len(Y.shape) != 2) or (Y.shape[0] != 1) : 
            raise ValueError('Incorrect shape for Y. Must be (1, N), N = number of samples.')
        
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same number of samples. In other words, X.shape[1] == Y.shape[1].')

    def _check_delays(self, X, delays):

        if (not isinstance(delays, list)) and (not isinstance(delays, int)):
            raise ValueError('Delays must be either an integer or a list.')

        if (isinstance(delays, list)) and (len(delays) != X.shape[0]):
            raise ValueError('The number of delays is wrong. Must satisfy len(delays) == X.shape[0].')

        if (isinstance(delays, list)) and (not all(isinstance(delay, int) for delay in delays)):
            raise ValueError('All elements of delays must be integers.')

    def _transform_delays(self, X, delays):
        """Transforms delays into a list if given as a single integer."""

        if isinstance(delays, int): delays = X.shape[0]*[delays]
        return delays

    def _extract_features_labels(self, dataset):
        pass

if __name__ == '__main__':

    X = np.random.rand(3,10)
    Y = np.random.rand(1,10)
    delays = 2

    my_model = GpDynModel(name='rnd_sys')
    my_model.load_data(X, Y, delays)
