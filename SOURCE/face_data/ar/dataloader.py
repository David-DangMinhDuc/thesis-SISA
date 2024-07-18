import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))
data = np.load(os.path.join(pwd, 'ar100_data.npy'), allow_pickle=True).reshape((1,))[0]
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

def load(indices, method='train'):
    if method == 'train':
        return X_train[indices], y_train[indices]
    elif method == 'test':
        return X_test[indices], y_test[indices]