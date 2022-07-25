"""
Train a neural network on lorenz coefficient manifold data
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 1/20/21; Last revision: 1/20/21
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import model_from_json
from timeit import default_timer as timer
from logistic_coefficient_map import logistic_coefficient_map


def ell1Norm(numpyVector):
    """Return the ell^1 norm of a numpy vector"""
    return np.sum(np.abs(numpyVector))

def count_MACC(layerDim, N, n=3):
    """Count number of MACC operations for a Lorenz NN with given layer dimensions"""
    d = [n] + layerDim + [n * N]
    return np.sum([d[i] * d[i+1] for i in range(len(d)-1)])

def layer_dims(alpha, N, n=3):
    """Return a vector of layer dimensions (uniform size) for the Lorenz system so that the resulting neural network has
    no more than N^2 MACC operations."""
    layerDim = int(alpha * N)  # dimension of each layer
    nLayer = 1 + int((n * N - alpha * n - alpha * n * N) / (N * alpha ** 2))
    return nLayer * [layerDim]



# Load saved training data and split into training/testing sets
dataFileName = 'lorenz_data_100_1e4.npz'  # numbers are N and training data size.
npzfile = np.load(dataFileName)
X = npzfile['arr_0']
Y = npzfile['arr_1']
nTrainingSet = npzfile['arr_2']
x_train, x_test = np.split(X, [nTrainingSet])
y_train, y_test = np.split(Y, [nTrainingSet])
imgSize = np.shape(y_train)[1]  # M = 3N + 1 is the dimension of the manifold embedding space i.e. dimension of the the images
N = imgSize // 3

# Define neural network model parameters and architecture
# Interior architecture #1: 3 = in ----> alpha * M ---->  alpha * M ----> ... ---->  alpha * M ---->  out = M.
# alpha = 0.56  # 5 layers
# alpha = 0.61  # 4 layers
alpha = 0.68  # 3 layers

modelLayerDims = layer_dims(alpha, N)
# modelLayerDims = [imgSize, 2 * imgSize, 3 * imgSize, 2 * imgSize, imgSize]
print(modelLayerDims)


modelActivation = 'relu'
modelKernelInit = 'he_uniform'

newModel = True
epochs = 100
batch_size = 128
model_optimizer = keras.optimizers.SGD(lr=1)
# model_optimizer = keras.optimizers.RMSprop(lr=0.01)
# model_optimizer = keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_loss = keras.losses.mean_squared_error
model_metrics = ['mae']
dropout_rate = 0

# build and train a network model
plt.close('all')
print("building new model")
model = Sequential()
input_shape = (3,)

# add input layer with dropout
model.add(Dense(imgSize, activation='linear',
                input_shape=input_shape))
if dropout_rate > 0:
    model.add(Dropout(dropout_rate))

# add hidden layers
for layerDim in modelLayerDims:
    # model.add(Dense(layerDim, activation='relu', kernel_initializer=relu_kernel, use_bias=True))
    model.add(Dense(layerDim, activation=modelActivation, kernel_initializer=modelKernelInit, use_bias=True))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

# add output layer
model.add(Dense(imgSize, activation='linear'))
model.compile(loss=model_loss,
              optimizer=model_optimizer,
              metrics=model_metrics)

# else:  # continue training an existing keras NN model
#     json_file = open('{0}.json'.format(loadModelFileName), 'r')  # load existing model
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     model.load_weights('{0}.h5'.format(loadModelFileName))  # load weights into new model
#     print("Loaded model from disk")
#     model.compile(loss=model_loss,
#                   optimizer=model_optimizer,
#                   metrics=model_metrics)
#
#     score = model.evaluate(x_test, y_test, verbose=0)
#     print('initial metrics:')
#     print("%s: %.10f" % (model.metrics_names[1], score[1]))

# Do a single round (100 epochs) of training
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('final metrics')
print('%s: %.10f' % (model.metrics_names[1], score[1]))
modelPredictionData = model.predict(x_test)
modelL1Error = np.array([ell1Norm(modelPredictionData[j, :] - y_test[j, :]) for j in range(np.shape(y_test)[0])])

# SERIALIZE MODEL TO JSON AND WEIGHTS TO HDF5
# saveModelFileName = 'logistic_model_{0}_{1}'.format(epochs, modelActivation)
# model_json = model.to_json()
# with open('{0}.json'.format(saveModelFileName), 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights('{0}.h5'.format(saveModelFileName))
# print("Saved model to disk")


# PLOTTING
# plot training history after initial transient states (epoch 5 onward)
plt.figure()
plt.plot(history.history['mae'][5:], label='mae_training')
plt.plot(history.history['val_mae'][5:], label='mae_val')
plt.legend()
plt.title('mae training history')
#
# # plot ell^1 error on validation data
# plt.figure()
# plt.scatter(x_test, modelL1Error, 1)
# plt.title('Epoch {0} (final): '.format(epochs))

# # plot true ell^1 norms of validation data
# plt.figure()
# plt.scatter(x_test, [ell1Norm(y_test[j, :]) for j in range(np.shape(x_test)[0])], 1)
# plt.title('L1 norm of validation data')


# # plot each coefficient for validation data
# plt.figure()
# for j in range(imgSize):
#     plt.scatter(x_test, y_test[:, j], 1, label='a_{0}'.format(j))
# plt.legend()
# plt.title('Coefficient Map Coordinates')

# # plot each coefficient error on validation data
# plt.figure()
# for j in range(imgSize):
#     plt.scatter(x_test, np.abs(y_test[:, j] - modelPredictionData[:, j]), 1, label='a_{0}'.format(j))
# plt.legend()
# plt.title('Coefficient Map Errors')
#
# # plot timestep errors
# if withTime:
#     plt.figure()
#     plt.scatter(x_test, y_test[:, -1])
#     plt.scatter(x_test, modelPredictionData[:, -1])
#     plt.show()

# # evaluation timing
# # NN eval
# timingData = -1 + 2 * np.random.rand(1000)
# start = timer()
# model.predict(timingData)
# end = timer()
# print(end - start)  # Time in seconds
#
# # recursion eval
# withTime = False  # specify if NN should try to learn the timestep as well or fix it
# if withTime:
#     eps = np.finfo(float).eps
#     imgSize = N + 1
#     TaylorCoefficientMap = lambda x: logistic_coefficient_map(x, N, lastCoefficientNorm=eps)
#
# else:
#     fixTau = 1
#     imgSize = N
#     TaylorCoefficientMap = lambda x: logistic_coefficient_map(x, N, tau=fixTau)

# start = timer()
# for x in timingData:
#     TaylorCoefficientMap(x)
# end = timer()
# print(end - start)  # Time in seconds


# start = timer()
# for x in timingData:
#     TaylorCoefficientMap(x)
# end = timer()
# print(end - start)  # Time in seconds
