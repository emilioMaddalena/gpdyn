import numpy as np 

def standardize_delays(Y, U, W, delays):
    """Transforms delays into a list if given as a single integer."""

    num_signals = 1 + U.shape[0] + W.shape[0] if (U and W) else 1 + U.shape[0] if (U and not W) else 1 

    #? maybe this
    #? return X.shape[0]*[delays] if isinstance(delays, int) else delays
    if isinstance(delays, int): delays = num_signals*[delays]
    return delays

def create_features_labels(Y, U, W, delays):

    max_delay = max(delays)
    num_signals = 1 + U.shape[0] + W.shape[0] if (U and W) else 1 + U.shape[0] if (U and not W) else 1 

    # Y, U and W will all compose the feature vector
    X = np.concatenate([Y, U, W], axis=0) if (U and W) else np.concatenate([Y, U], axis=0) if (U and not W) else Y

    # shift signals by every delay value, store all together
    #? could also be done by copying X a couple of times
    #? and passing a several delays at once to np.roll
    XX = [] 
    for sig_idx in range(num_signals):
        for delay in range(1, delays[sig_idx]+1):
            XX.append(np.roll(X[sig_idx,:], delay))
    XX = np.stack(XX) # shape = (\sum{signal*delay}, num_data)

    # cut-off the first couple of values (needed due to np.roll)
    features = XX[:,max_delay:]
    labels = Y[:,max_delay:]

    return features, labels