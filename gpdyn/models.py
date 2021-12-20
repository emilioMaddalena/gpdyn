import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities import print_summary, set_trainable

from gpdyn.checks import check_data_shapes, check_delays
from gpdyn.process import standardize_delays, create_features_labels

class GpDynModel:
    """
    An auto-regressive Gaussian process model with exogenous inputs.
    """

    def __init__(self, name=None):
        
        self.Y = None
        self.U = None
        self.W = None
        self.name = name

        self._data_was_loaded = False
        self._model_was_trained = False

    def load_data(self, Y, U=None, W=None):
        """
        Args:
            Y: np.array of shape (1, N). N is the number of data.
            U: np.array of shape (U, N). U is the number of exogenous signals and N is the number of data.
            W: np.array of shape (W, N). W is the number of exogenous signals and N is the number of data.
        """

        Y = np.array(Y).reshape(1,-1)
        check_data_shapes(Y, U, W)

        self.Y = Y
        self.U = U
        self.W = W
        self._data_was_loaded = True

    def train_model(self, delays):

        if not self._data_was_loaded: 
            print('Call load_data() before calling train_model().')
            return

        check_delays(self.Y, self.U, self.W, delays)
        delays = standardize_delays(self.Y, self.U, self.W, delays)
        self.delays = delays

        self.feats, self.labels = create_features_labels(self.Y, self.U, self.W, self.delays)
        data = (tf.convert_to_tensor(self.feats.T, "float64"), tf.convert_to_tensor(self.labels.T, "float64"))

        kernel = gpflow.kernels.Matern52()
        gp = gpflow.models.GPR(data, kernel=kernel)
        
        #print("\nPre-training:")
        #print_summary(gp)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(gp.training_loss, 
                     gp.trainable_variables, 
                     tol=1e-8, 
                     method='l-bfgs-b',
                     options=dict(maxiter=1000))
        #print("\nPost-training:")
        #print_summary(gp)

        self.gp = gp
        self._model_was_trained = True

    def test_model(self, Y, U=None, W=None, horizon=1):

        if not self._model_was_trained: 
            print('Call train_model() before calling test_model().')
            return

        feats, labels = create_features_labels(Y, U, W, self.delays)
        
        horizon += 1 
        max_label_delay = self.delays[0]

        # rouding the test_dataset to a multiple of the horizon
        N_data = int(feats.shape[1]/horizon) * horizon
        feats = feats[:,:N_data]
        labels = labels[:,:N_data].reshape(1,-1)
 
        feats_open_loop = np.empty(feats.shape)
        mean_pred = np.empty((N_data))
        var_pred = np.empty((N_data))

        # start fresh and insert predictions later
        feats_open_loop = np.copy(feats)
        # predict in batches
        for step in range(horizon):
            m, v = self.gp.predict_f(feats_open_loop[:,step::horizon].T) 
            mean_pred[step::horizon] = m.numpy().reshape(-1,)
            var_pred[step::horizon] = v.numpy().reshape(-1,)
            # propagate forward the prediction, not violating the horizon window
            for label_delay in range(max_label_delay): 
                if step+1+label_delay < horizon: 
                    feats_open_loop[label_delay,step+label_delay+1::horizon] = mean_pred[step::horizon]

        return feats, labels, mean_pred, var_pred

    def predict(self):
        pass

if __name__ == '__main__':

    X = np.stack([np.arange(1.0,10.0+1), np.arange(101.0,110.0+1)])
    Y = np.arange(1.0,10.0+1).reshape(1,-1)
    delays = 2

    my_model = GpDynModel(name='rnd_sys')
    my_model.load_data(Y)

    my_model.train_model(delays)

    Z = np.arange(101.0,130.0+1).reshape(1,-1)
    my_model.test_model(Z, horizon=4)
