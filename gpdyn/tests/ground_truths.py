import numpy as np

class SimpleAr_2D:
    """y[t+1] = 0.9*y[t] + sin(y[t-1])"""

    def __init__(self):

        self.dim = 2
        self.ymin = -10.
        self.ymax = 10.
        
    def __call__(self, y):

        y = np.array(y)

        ground_truth = lambda y_t, y_tt: 0.9*y_t + 0.5*np.sin(y_tt)
        
        # handle the scalar and vectorized cases
        if (len(y.shape) == 0) or (len(y.shape) == 1): y_t, y_tt = (y[0], y[1])
        elif len(y.shape) == 2: y_t, y_tt = (y[0,:], y[1,:])
        
        return ground_truth(y_t, y_tt)

    def get_data(self, y_init, t_sim, noise_sigma=None):

        y = np.empty(t_sim+1)
        
        y[0] = y_init
        y[1] = self.__call__([y_init, 0])
        
        for t in range(1,t_sim): y[t+1] = self.__call__([y[t], y[t-1]])
        if noise_sigma: y += np.random.normal(0, noise_sigma, y.shape)

        return y