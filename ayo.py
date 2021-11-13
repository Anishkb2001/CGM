from keras.layers import Dense, Dropout, BatchNormalization, Normalization
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

t_data = sio.loadmat('/Users/anishkumarbadri/Desktop/Research/new/Research-selected/input_samples_2.mat')
out_data = sio.loadmat('/Users/anishkumarbadri/Desktop/Research/new/Research-selected/output.mat')
n_train = 403
x = t_data['final']
y = out_data['out_i']

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
train_x, test_x, train_y,test_y = train_test_split(x, y, test_size = 0.31, random_state=42)


def baseline_model():
    in_dims = 1463
    model = Sequential()
    model.add(Dense(128, input_dim=in_dims, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(model=baseline_model, epochs=300, batch_size=32, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(train_x,train_y)
prediction = estimator.predict(test_x)
print(r2_score(test_y, prediction))
pred_index =list(range(403, 585))
plt.scatter(pred_index, prediction)
plt.scatter(pred_index,test_y)
plt.show()






