import keras
from numpy import loadtxt
from keras.layers import Dense, Dropout, BatchNormalization, Normalization
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.io as sio
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

t_data = sio.loadmat('/Users/anishkumarbadri/Desktop/Research/new/Research-selected/input_samples.mat')
out_data = sio.loadmat('/Users/anishkumarbadri/Desktop/Research/new/Research-selected/output.mat')
n_train = 403
x = t_data['final']
y = out_data['out_i']

x_mean = x.mean(axis = 1 )
x_std = x.std(axis = 1 )
x_mean = np.reshape(x_mean, (585,1))
x_std = np.reshape(x_std, (585,1))
x = (x-x_mean)/x_std

# trans = MinMaxScaler()
# data = trans.fit_transform(x)
train_x, test_x = x[0:n_train, :], x[n_train:, :]
train_y, test_y = y[0:n_train], y[n_train:]

drop = 0.5


in_dims = 1463

model = Sequential()
# model.add(BatchNormalization(axis=1,momentum=0.99,epsilon=0.001,input_shape=(in_dims,)))
model.add(Dense(128, input_dim=in_dims, kernel_initializer='normal', activation='relu'))
model.add(Dropout(drop))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(drop))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_x, train_y, epochs=300, batch_size=10)

y_pred = model.predict(test_x)
print(mean_squared_error(test_y, y_pred))
print(r2_score(test_y, y_pred))



# pred_index =list(range(n_train, 585))
# plt.scatter(pred_index, y_pred)
# plt.scatter(pred_index,test_y)
# plt.show()


# estimator = KerasRegressor(build_fn=basemodel, epochs=900, batch_size=22, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, train_x, train_y, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# model.fit(train_x, train_y, epochs=150, batch_size=10)
# _, accuracy = model.evaluate(train_x, train_y)
# print('Accuracy: %.2f' % (accuracy*100))
