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

        #? maybe this
        #? return X.shape[0]*[delays] if isinstance(delays, int) else delays

        if isinstance(delays, int): delays = X.shape[0]*[delays]
        return delays

    def _extract_features_labels(self, dataset):
        '''
        Given a dataset of time-series composed of different groups (disconnected experiments), 
        constructs X and y to be used by the Gaussian processes. This is done by processing the 
        time-series according to the model.feat_names and model.feat_delays.

        Args: 
            dataset: Pandas dataframe. Time-series of different signals separated in different groups.
        Return:
            X (np array shape (N,M)): N is the number of data for the GP to be trained on, M is  
               the dimension of the X space. X is the collection of inputs.
            y (np array shape (N,1)): N is the number of data. y is the collection of outputs 
               associated with X.
        '''
        
        label_idx = self.feat_names.index(self.label_name)

        flat_delays = [delay for sublist in self.feat_delays for delay in sublist]
        max_delay = max(flat_delays)
        
        X_list = []
        y_list = []

        for batch_num in dataset['batch'].unique():
            batch = dataset[dataset['batch'] == batch_num]
            
            # collecting all features within a batch, with all corresponding delays
            xx = [] 
            for feat_number, feat_name in enumerate(self.feat_names): 
                # Delay and accumulate past features
                for delay in self.feat_delays[feat_number]:
                    # shifting based on delay
                    x = np.roll(np.array(batch[feat_name]), delay, axis=0) 
                    x = list(x.flatten())
                    xx.append(x)

            xx = np.vstack(xx).transpose()     
            X_list.append(xx[max_delay:-1,:])  
            y_list.append(xx[max_delay+1:, label_idx])

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        return X, y

#     def _remove_close_data(self, X, y, threshold=0):
#         '''
#         For all points, calculate the their distance to their closest neighbor.
#         Then, remove from the dataset (features and labels) all points whose
#         distance is smaller than the given threshold.
#         '''
#         if threshold > 0:
#             N_samp = X.shape[0] 
#             dists  = np.empty((N_samp,))
#             for i in range(N_samp):
#                 x = X[i,:].reshape(1,-1)
#                 z = np.delete(X, i, 0)
#                 dists[i] = np.min(pairwise_distances(z, x))
#             mask = dists >= threshold
#             X_new = X[mask,:]
#             y_new = y[mask]
#             print("Initial dataset contained {} data.".format(X.shape[0]))
#             print("Spaced dataset contains {} data.".format(X_new.shape[0]))
#             return X_new, y_new
#         else: return X, y

#     def _assemble_kernel_function(self, kernel, ard, N_feat):

#         if isinstance(kernel, list) and len(kernel) != N_feat: 
#             raise ValueError('If passed as a list, the kernel parameter must be of length either N_feat.')
        
#         # One single kernel
#         if isinstance(kernel, str):
#             if ard: len_sca = np.random.uniform(1, 10, self.N_feat)
#             else: len_sca = np.random.uniform(1, 10, 1)
#             # which?
#             if   kernel == 'SE': ker = gpflow.kernels.SquaredExponential(lengthscales=len_sca)
#             elif kernel == 'RQ': ker = gpflow.kernels.RationalQuadratic(lengthscales=len_sca)
#             elif kernel == 'LI': ker = gpflow.kernels.Linear(variance=len_sca)

#         # Multiple kernels (one per input dimension)
#         elif isinstance(kernel, list):
#             for current_kernel in ['SE', 'RQ', 'LI']:
#                 # Extract dimensions where that kernel is used
#                 dims = [idx for idx, elem in enumerate(kernel) if elem == current_kernel]
#                 num_dims = len(dims)
#                 # If not empty
#                 if num_dims != 0: 
#                     if ard: len_sca = np.random.uniform(1, 10, num_dims)
#                     else: len_sca = np.random.uniform(1, 10, 1)
#                     # which?
#                     if   current_kernel == 'SE': k = gpflow.kernels.SquaredExponential(lengthscales=len_sca, active_dims=dims)
#                     elif current_kernel == 'RQ': k = gpflow.kernels.RationalQuadratic(lengthscales=len_sca, active_dims=dims)
#                     elif current_kernel == 'LI': k = gpflow.kernels.Linear(variance=len_sca, active_dims=dims)
#                     # sum all kernels
#                     try: ker = ker + k
#                     except: ker = k
#         return ker

#     def _scale_dataset(self, scale_x, scale_y, X, y):
#         '''
#         scaled_data = (data - mean) / stdv
#         '''
#         self.scalers = {'scaled_x': scale_x, 'scaled_y': scale_y}
#         if scale_x:    
#             x_scaler_mean = np.mean(X, axis=0)
#             x_scaler_stdv = np.sqrt(np.var(X, axis=0))
#             X = (X - x_scaler_mean) / x_scaler_stdv
#             self.scalers['x_scaler_mean'] = x_scaler_mean
#             self.scalers['x_scaler_stdv'] = x_scaler_stdv
#         if scale_y:
#             y_scaler_mean = np.mean(y, axis=0)
#             y_scaler_stdv = np.sqrt(np.var(y, axis=0))
#             y = ((y - y_scaler_mean) / y_scaler_stdv)
#             self.scalers['y_scaler_mean'] = y_scaler_mean
#             self.scalers['y_scaler_stdv'] = y_scaler_stdv
#         return X, y.reshape(-1,1)
    
#     def _scale_X(self, scale_x, X):
#         if scale_x:
#             x_scaler_mean = self.scalers['x_scaler_mean']
#             x_scaler_stdv = self.scalers['x_scaler_stdv']
#             return (X - x_scaler_mean) / x_scaler_stdv
#         else:
#             return X

#     def _inverse_scale_y(self, scale_y, y):
#         '''
#         data = (scaled_data * stdv) + mean
#         '''
#         if scale_y:    
#             y_scaler_mean = self.scalers['y_scaler_mean']
#             y_scaler_stdv = self.scalers['y_scaler_stdv']
#             return (y * y_scaler_stdv) + y_scaler_mean
#         else:
#             return y

#     def train_model(self, train_dataset, kernel, ard, linear_mean, 
#                     scale_x=True, scale_y=False, hyper_prior=False, 
#                     dist_thresh=0, noise_var=0.01, jitter=1e-6, max_iter=1000, tol=1e-8):
#         """
#         Trains the GP model on train_dataset through either MLE or MAP.

#         Args: 
#             train_dataset (pandas dataframe): Time-series of different signals separated in different groups.
#             kernel (string): Selects the GP kernel. 'RQ', 'SE' or 'LI'.
#             ard (bool): Enables or disables Automatic Relevance Determination, i.e., having one lengthscale per X dimension. 
#             hyper_prior (bool): Enables or disables the use of hyperparameter priors, which are hardcoded in this method.
#             linear_mean (bool): Enables or disables the use of a linear mean function, i.e., a semi-parametric model.
#             noise_var (float): Specifies the initial guess for the white noise variance.
#             jitter (float): Specifies the default kernel matrix diagonal jitter to be used by GPflow.
#             max_iter (int): Bounds the maximum number of iterations when optimizing the hyperparameters.
#             tol (float): Specifies the solver tolerance when optimizing the hyperparameters.
#         """

#         training_options = {'kernel': kernel,
#                             'ard': ard,
#                             'linear_mean': linear_mean,
#                             'hyper_prior': hyper_prior,
#                             'dist_thresh': dist_thresh,
#                             'noise_var': noise_var,
#                             'jitter': jitter,
#                             'max_iter': max_iter,
#                             'tol': tol}
        
#         X, y = self._extract_features_labels(train_dataset)
#         X, y = self._remove_close_data(X, y, dist_thresh)
#         X, y = self._scale_dataset(scale_x, scale_y, X, y)

#         N_data = y.shape[0]
#         N_feat = X.shape[1]

#         # TODO still broken
#         # if hyper_prior: 
#         #     # See: https://homepage.divms.uiowa.edu/~mbognar/applets/gamma.html
#         #     # ker.lengthscales.prior = tfp.distributions.Gamma(10.0 * np.ones((self.N_feat,1)), 3.0 * np.ones((self.N_feat,1)))
#         #     ker.lengthscales.prior = tfp.distributions.Gamma(np.array([5., 5., 5., 30., 500., 20., 30., 30., 20.]),
#         #                                                      np.array([3., 3., 3., 4., 5., 10., 10., 10., 10.]))
#         data  = (tf.convert_to_tensor(X, "float64"), 
#                  tf.convert_to_tensor(y, "float64"))
#         ker = self._assemble_kernel_function(kernel, ard, N_feat)
#         if linear_mean: mean = gpflow.mean_functions.Linear(A=np.ones((N_feat,1)), b=1.) 
#         else: mean = None
#         gp = gpflow.models.GPR(data, kernel=ker, mean_function=mean) 

#         gp.likelihood.variance.assign(noise_var)
#         set_trainable(gp.likelihood.variance, False)
#         gpflow.config.set_default_summary_fmt("notebook")
#         gpflow.config.set_default_jitter(jitter)
        
#         opt = gpflow.optimizers.Scipy()
#         opt.minimize(gp.training_loss, 
#                      gp.trainable_variables, 
#                      tol=tol, 
#                      method='l-bfgs-b',
#                      options=dict(maxiter=max_iter))
        
#         # compute training error
#         mean, _ = gp.predict_f(X)
#         y_pred = self._inverse_scale_y(scale_y, mean)
#         RMSE_train = np.sqrt(np.average(np.square(y - y_pred)))

#         self.scale_x = scale_x
#         self.scale_y = scale_y
#         self.RMSE_train = RMSE_train
#         self.training_options = training_options
#         self.N_data = N_data
#         self.N_feat = N_feat
#         self.X = X 
#         self.y = y
#         self.gp = gp
#         self.log_marginal_likelihood = float(self.gp.log_marginal_likelihood())
        
#     def test_model(self, test_dataset, horizon, save_data=False):
#         '''
#         Predicts the response variable on a single group, i.e., a single time-series.
#         Assumes only one group is given in test_dataset.

#         Args: 
#             test_dataset (pandas dataframe):
#             horizon (int): 
#         Returns:
#             X_test (np array): 
#             y_test (np array): 
#             y_pred (np array):
#             RMSE (float):
#         '''
#         X_test, y_test = self._extract_features_labels(test_dataset)

#         # temporary augmentation. + 1 for the 'support points'
#         horizon = horizon + 1

#         # rouding the test_dataset to a multiple of the horizon
#         N_data = int(len(X_test) / horizon) * horizon #? not needed maybe
#         #print('N_data = {}'.format(N_data))
#         X_test = X_test[:N_data,:]
#         y_test = y_test[:N_data]
#         y_test = y_test.reshape(-1,1)

#         X_OL   = np.empty(X_test.shape)
#         y_pred = np.empty((N_data, 1))
#         y_var  = np.empty((N_data, 1))

#         label_idx = self.feat_names.index(self.label_name)
#         max_label_delay = max(self.feat_delays[label_idx])

#         # start fresh
#         X_OL = np.copy(X_test)

#         for i in range(horizon):
#             # scale features properly and predict
#             # OLD: X = self.Xscaler.transform(X_OL[i::horizon,:]) # stopped here
#             # OLD: y_pred[i::horizon], _ = self.yscaler.inverse_transform(self.gp.predict_f(X))
#             # OLD: _, y_var[i::horizon] = self.gp.predict_f(X)
#             X = self._scale_X(self.scale_x, X_OL[i::horizon,:]) # stopped here
#             mean, var = self.gp.predict_f(X)
#             y_pred[i::horizon] = self._inverse_scale_y(self.scalers['scaled_y'], mean)
#             y_var[i::horizon] = var #! inverse scale it?

#             # propagate forward the prediction, not violating the refresh @ every horizon
#             for d in range(max_label_delay + 1): 
#                 if i+1+d < horizon:
#                     #print('Here go i = {} and d = {}'.format(i,d)) # correct!
#                     X_OL[i+d+1::horizon, label_idx+d] = y_pred[i::horizon].reshape(-1,)

#         RMSE = np.sqrt(np.average(np.square(y_test - y_pred)))
#         self.RMSE_test = RMSE

#         if save_data: 
#             self.X_test = X_test
#             self.y_test = y_test
#             self.y_pred = y_pred
#             self.y_var  = y_var
#             self.horizon = horizon - 1 
#             #self.horizon = horizon
#         return X_test, y_test, y_pred, y_var, RMSE

#     def inspect_lengthscales(self):
#         '''
#         Prints the features, delays and lengthscales of a model.
#         '''

#         print('\nFeatures: \n', self.feat_names)
#         print('\nDelays: \n', self.feat_delays)
#         # one kernel
#         try: 
#             print('\nLengthscales: \n', self.gp.kernel.lengthscales.numpy())
#         # multiple kernels
#         except: 
#             for kernel in self.gp.kernel.kernels:
#                 if kernel.name != 'linear': print('\nLengthscales for', kernel.name, ':', kernel.lengthscales.numpy())

#     def plot_data_distribution(self, xlim):
#         '''
#         Prints the features, delays and lengthscales of a model.
#         '''
#         dists_new = np.empty((self.X.shape[0],))
#         for i in range(self.X.shape[0]):
#             x = self.X[i,:].reshape(1,-1)
#             z = np.delete(self.X, i, 0)
#             dists_new[i] = np.min(pairwise_distances(z, x))
#         pl = sns.displot(dists_new, bins=self.N_data//10)
#         pl.set(xlim=xlim)

# def model_to_json(model):
#     '''
#     Serializes a model and save its parameters.

#     Args: 
#         model (model): The model object to be saved to a json file.
#         filepath (str): The path and name of the file to be created.
#     '''
    
#     dict = {}

#     dict['label_name']  = model.label_name
#     dict['feat_names']  = model.feat_names
#     dict['feat_delays'] = model.feat_delays
#     dict['N_feat'] = model.N_feat
#     dict['N_data'] = model.N_data 
#     dict['log_marginal_likelihood'] = model.log_marginal_likelihood
#     dict['RMSE_test']  = model.RMSE_test
#     dict['RMSE_train'] = model.RMSE_train
#     dict['training_options'] = model.training_options

#     # NEW
#     dict['scalers'] = {}
#     dict['scalers']['scaled_x'] = model.scalers['scaled_x']
#     dict['scalers']['scaled_y'] = model.scalers['scaled_y']
#     if 'x_scaler_mean' in model.scalers: dict['scalers']['x_scaler_mean'] = model.scalers['x_scaler_mean'].tolist()
#     if 'y_scaler_mean' in model.scalers: dict['scalers']['y_scaler_mean'] = model.scalers['y_scaler_mean'].tolist()
#     if 'x_scaler_stdv' in model.scalers: dict['scalers']['x_scaler_stdv'] = model.scalers['x_scaler_stdv'].tolist()
#     if 'y_scaler_stdv' in model.scalers: dict['scalers']['y_scaler_stdv'] = model.scalers['y_scaler_stdv'].tolist()
    
#     # Extract all GP parameters and convert them into lists (needed to serialize the dictionary)
#     param_dic = gpflow.utilities.parameter_dict(model.gp)
#     for key, value in param_dic.items():
#         try: param_dic[key] = value.numpy().tolist()
#         except: ValueError('Problem converting GP parameters into a serializable dictionary.')
#     dict['gp_params']  = param_dic
    
#     dict['X'] = model.X.tolist()
#     dict['y'] = model.y.tolist()

#     if hasattr(model, 'X_test'):  dict['X_test']  = model.X_test.tolist()
#     if hasattr(model, 'y_test'):  dict['y_test']  = model.y_test.tolist()
#     if hasattr(model, 'y_pred'):  dict['y_pred']  = model.y_pred.tolist()
#     if hasattr(model, 'y_var'):   dict['y_var']   = model.y_var.tolist()
#     if hasattr(model, 'horizon'): dict['horizon'] = model.horizon

#     import os.path

#     # Find an appropriate number for the model
#     idx = 0
#     while os.path.isfile('./saved_models/' + model.label_name + '_model_{}.json'.format(idx)): idx += 1
#     filepath = './saved_models/' + model.label_name + '_model_{}.json'.format(idx)

#     # Create json and save to file
#     json_txt = json.dumps(dict, indent=4)
#     with open(filepath, 'w') as file:
#         file.write(json_txt)

#     return filepath

# def model_from_json(filepath):
#     '''
#     De-serializes a model and save its parameters.

#     Args: 
#         filepath (str): The path and name of the file to be created.
#     Returns:
#         model (model): The model object loaded from a json file.
#     '''
#     with open(filepath, 'r') as file:
        
#         dict = json.load(file)

#         model = Model(dict['feat_names'], dict['feat_delays'], dict['label_name'])

#         model.X = np.array(dict['X'])
#         model.y = np.array(dict['y'])

#         if "X_test" in dict: model.X_test = np.array(dict['X_test'])
#         if "y_test" in dict: model.y_test = np.array(dict['y_test'])
#         if "y_pred" in dict: model.y_pred = np.array(dict['y_pred'])
#         if "y_var"  in dict: model.y_var  = np.array(dict['y_var'])
#         if "horizon" in dict: model.horizon = np.array(dict['horizon'])
        
#         model.RMSE_test = dict['RMSE_test']
#         model.RMSE_train = dict['RMSE_train']
#         model.N_data = dict['N_data']
#         model.N_feat = dict['N_feat']

#         model.scalers = {'scaled_x': dict['scalers']['scaled_x'], 
#                          'scaled_y': dict['scalers']['scaled_y']}
#         if dict['scalers']['scaled_x']:    
#             model.scalers['x_scaler_mean'] = np.array(dict['scalers']['x_scaler_mean'])
#             model.scalers['x_scaler_stdv'] = np.array(dict['scalers']['x_scaler_stdv'])
#         if dict['scalers']['scaled_y']:    
#             model.scalers['y_scaler_mean'] = np.array(dict['scalers']['y_scaler_mean'])
#             model.scalers['y_scaler_stdv'] = np.array(dict['scalers']['y_scaler_stdv'])
        
#         # NEW:
#         model.scale_x = dict['scalers']['scaled_x']
#         model.scale_x = dict['scalers']['scaled_y']

#         model.training_options = dict['training_options']

#         data  = (tf.convert_to_tensor(model.X, "float64"), 
#                  tf.convert_to_tensor(model.y, "float64"))

#         # Re-building the GP
#         ker = model._assemble_kernel_function(dict['training_options']['kernel'], 
#                                               dict['training_options']['ard'],
#                                               dict['N_feat'])

#         if dict['training_options']['linear_mean']: mean = gpflow.mean_functions.Linear(A=np.ones((model.N_feat,1)), b=1.) 
#         else: mean = None

#         gp = gpflow.models.GPR(data, kernel=ker, mean_function=mean) 

#         params = dict['gp_params']
#         gpflow.utilities.multiple_assign(gp, params)

#         model.gp = gp
#         model.log_marginal_likelihood = float(model.gp.log_marginal_likelihood())
        
#         print("\nA GP model was loaded for '{0}'. Attained log marginal likelihood: {1} ".format(model.label_name, model.log_marginal_likelihood))

#         return model

# def log_to_csv(filepath):
#     '''
#     Logs the details of the model found at filepath into a csv file.

#     Args: 
#         filepath (string):
#     ''' 

#     with open(filepath, 'r') as file:
    
#         dict = json.load(file)
        
#         idx1 = filepath.rfind('/') 
#         idx2 = filepath.rfind('.') 
#         filename = filepath[idx1+1:idx2]

#         for element in dict['training_options']['kernel']:
#             try: ker = ker + "," + element
#             except: ker = element
        
#         results = {'model': filename,
#                    'num_data': [dict['N_data']],
#                    'RMSE_train': [round(dict['RMSE_train'], 3)],
#                    'RMSE_test': [round(dict['RMSE_test'], 3)],
#                    'log_marg_likelihood': [round(dict['log_marginal_likelihood'], 3)],
#                    'noise_variance': [round(dict['gp_params']['.likelihood.variance'], 3)],
#                    'feat_names': [dict['feat_names']],
#                    'feat_delays': [dict['feat_delays']],
#                    'kernel': ker,#dict['training_options']['kernel'],
#                    'ard': dict['training_options']['ard'],
#                    'hyper_prior': dict['training_options']['hyper_prior'],
#                    'linear_mean': dict['training_options']['linear_mean'],
#                    'jitter': dict['training_options']['jitter'],
#                    'max_iter': dict['training_options']['max_iter'],
#                    'tol': dict['training_options']['tol']}
        
#     struct = pd.DataFrame(results)
#     logfile = "./saved_models/info.csv"

#     try:
#         pd.read_csv(logfile)
#         struct.to_csv(logfile, mode='a', header=False, index=False)
#         #print('File found!')          

#     except:
#         struct.to_csv(logfile, mode='a', index=False)
#         #print('No file found...')

if __name__ == '__main__':

    X = np.random.rand(3,10)
    Y = np.random.rand(1,10)
    delays = 2

    my_model = GpDynModel(name='rnd_sys')
    my_model.load_data(X, Y, delays)

# %%
