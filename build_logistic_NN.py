from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.models import model_from_json
# from keras import backend as K
# from keras import optimizers
from timeit import default_timer as timer
# from logistic_coefficient_map import logistic_coefficient_map





# SET TRAINING PARAMETERS
# loadModelFileName = 'logistic_model'  # name of saved filename for a NN model to load
training_orbit_fileName = 'taylor_logistic_training_test.npz'  # [N, x_train, y_train, x_test, y_test]

# LOAD SAVED TRAINING AND TESTING DATA
npzfile = np.load(dataFileName)
N = npzfile['arr_0']
x_train = npzfile['arr_1']
y_train = npzfile['arr_2']
x_test = npzfile['arr_3']
y_test = npzfile['arr_4']
imgSize = np.shape(y_train)[1]

# Interior architecture #1: N ----> in, 3N, 4N, 3N, out ---->
modelLayerDims = [j * imgSize for j in [3, 4, 3]]
modelActivation = 'relu'
modelKernelInit = 'he_uniform'
# modelKernelInit = 'zero'


# Architecture #2: N ----> N^2, 2N^2, N^2, N ---->
# modelLayerDims = [imgSize**2, 2*imgSize**2, imgSize**2, imgSize]

# Architecture #3: N ----> N, N,...., N, N ----> THIS IS NOT SO BAD
# modelLayerDims = [j * imgSize for j in [1 for i in range(imgSize)]] + [imgSize]

newModel = True
epochs = 50
batch_size = 128
model_optimizer = keras.optimizers.SGD(lr=0.1)
# model_optimizer = keras.optimizers.RMSprop(lr=0.01)
# model_optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_loss = keras.losses.mean_squared_error
model_metrics = ['mae']
dropout_rate = 0

saveModelFileName = 'logistic_model_{0}_{1}'.format(epochs, modelActivation)

# BUILD A NEW MODEL OR IMPORT AN EXISTING MODEL
# if newModel:  # build a new keras NN model
plt.close('all')
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

def ell1Norm(coefs):
    return sum(abs(coefs))

modelL1Error = np.array([ell1Norm(modelPredictionData[j, :] - y_test[j, :]) for j in range(np.shape(y_test)[0])])


# # SERIALIZE MODEL TO JSON AND WEIGHTS TO HDF5
# model_json = model.to_json()
# with open('{0}.json'.format(saveModelFileName), 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights('{0}.h5'.format(saveModelFileName))
# print("Saved model to disk")

# plot training history after initial transient states (epoch 5 onward)
plt.figure()
plt.plot(history.history['mae'][5:], label='mae_training')
plt.plot(history.history['val_mae'][5:], label='mae_val')
plt.legend()
plt.title('mae training history')

# plot ell^1 error on validation data
plt.figure()
plt.scatter(x_test, modelL1Error, 1)
plt.title('Epoch {0} (final): '.format(epochs))

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

# plot each coefficient error on validation data
plt.figure()
for j in range(imgSize):
    plt.scatter(x_test, np.abs(y_test[:, j] - modelPredictionData[:, j]), 1, label='a_{0}'.format(j))
plt.legend()
plt.title('Coefficient Map Errors')

# plot timestep errors
plt.figure()
plt.scatter(x_test, y_test[:, 0])
plt.scatter(x_test, modelPredictionData[:, 0])
plt.show()



