def check_data_shapes(Y, U, W):

    if (len(Y.shape) != 2) or (Y.shape[0] != 1) : 
        raise ValueError('Incorrect shape for Y. Must be (1, N), N = number of samples.')
    
    if U and ((len(U.shape) != 2) or (U.shape[1] != Y.shape[1])): 
        raise ValueError('Incorrect shape for U. Must be (U, N), U = number of input signals and N = number of samples.')
    
    if W and ((len(W.shape) != 2) or (W.shape[1] != Y.shape[1])): 
        raise ValueError('Incorrect shape for W. Must be (W, N), W = number of exogenous signals and N = number of samples.')
    
def check_delays(Y, U, W, delays):

    num_signals = 1 + U.shape[0] + W.shape[0] if (U and W) else 1 + U.shape[0] if (U and not W) else 1 

    if (not isinstance(delays, list)) and (not isinstance(delays, int)):
        raise ValueError('Delays must be either an integer or a list.')

    if (isinstance(delays, int)) and (delays <= 0):
        raise ValueError('delays must be an integer > 0 or a list of such integers.')

    if (isinstance(delays, list)) and ((not all(isinstance(delay, int)) or (delay <= 0) for delay in delays)):
        raise ValueError('delays must be an integer > 0 or a list of such integers.')

    if (isinstance(delays, list)) and (len(delays) != num_signals):
        raise ValueError('The number of delays is wrong. Must be = 1 + U.shape[0] + W.shape[0].')

