"""
Code used to build neural network models for the logistic example in the paper:
"A deep learning approach to efficient parameterization of invariant manifolds for continuous  dynamical systems"

    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 3/9/2020;
"""


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# from keras.models import model_from_json
# from keras import backend as K
# from keras import optimizers
# from timeit import default_timer as timer
# from logistic_coefficient_map import logistic_coefficient_map


def ell1Norm(coefs):
    return sum(abs(coefs))



class CustomMSELoss(keras.losses.Loss):
    def __init__(self, name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        return mse






def mse_custom(y_true, y_pred):

    return tf.math.reduce_mean(tf.square(y_true - y_pred))

# Define training orbit data and set architecture
training_orbit_fileName = 'taylor_logistic_training_test.npz'  # [N, x_train, y_train, x_test, y_test]


# LOAD SAVED TRAINING AND TESTING DATA
npzfile = np.load(training_orbit_fileName)
N = npzfile['arr_0']
x_train = npzfile['arr_1']
y_train = npzfile['arr_2']
x_test = npzfile['arr_3']
y_test = npzfile['arr_4']
imgSize = np.shape(y_train)[1]

# Model architecture
modelLayerDims = [j * imgSize for j in [3, 4, 3]]  # test architecture: (n=1) -input--> in, 3N, 4N, 3N, (N+1)-output-->
modelActivation = 'relu'
modelKernelInit = 'he_uniform'
dropout_rate = 0


# training parameters
epochs = 50
batch_size = 128
model_optimizer = keras.optimizers.SGD(lr=0.1)
# model_optimizer = keras.optimizers.RMSprop(lr=0.01)
# model_optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_loss = keras.losses.mean_squared_error
# model_loss = CustomMSE()
model_metrics = ['mae', mse_custom, CustomMSEMetric]
saveModelFileName = 'taylor_logistic_test'


print("building new model")
model = Sequential()
input_shape = (1,)

# add input layer with dropout
model.add(Dense(imgSize, activation='linear',
                input_shape=input_shape))
if dropout_rate > 0:
    model.add(Dropout(dropout_rate))

# add hidden layers
for layerDim in modelLayerDims:
    model.add(Dense(layerDim, activation=modelActivation, kernel_initializer=modelKernelInit, use_bias=True))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

# add output layer
model.add(Dense(imgSize, activation='linear'))
model.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=model_metrics)

# Do a single round of training
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('final metrics')
print('%s: %.10f' % (model.metrics_names[1], score[1]))
modelPredictionData = model.predict(x_test)





# plot training history after initial transient states (epoch 5 onward)
plt.close('all')
plt.figure()
plt.plot(history.history['mae'][5:], label='mae_training')
plt.plot(history.history['val_mae'][5:], label='mae_val')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['mse_custom'][5:], label='mse_training')
plt.plot(history.history['val_mse_custom'][5:], label='mse_val')
plt.legend()
plt.show()

# plot ell^1 error on validation data
modelL1Error = np.array([ell1Norm(modelPredictionData[j, :] - y_test[j, :]) for j in range(np.shape(y_test)[0])])
plt.figure()
plt.scatter(x_test, modelL1Error, 1)
plt.title('Epoch {0} (final): '.format(epochs))
plt.show()


# plot each coefficient error on validation data
plt.figure()
for j in range(imgSize):
    plt.scatter(x_test, np.abs(y_test[:, j] - modelPredictionData[:, j]), 1, label='a_{0}'.format(j))
plt.legend()
plt.title('Coefficient Map Errors')
plt.show()

# plot timestep errors
plt.figure()
plt.title('what is this')
plt.scatter(x_test, y_test[:, 0])
plt.scatter(x_test, modelPredictionData[:, 0])
plt.show()
