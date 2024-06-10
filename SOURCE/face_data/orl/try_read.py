import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))
data = np.load(os.path.join(pwd, 'orl40_data.npy'), allow_pickle=True).reshape((1,))[0]
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
