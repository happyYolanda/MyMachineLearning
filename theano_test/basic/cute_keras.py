import numpy as np

from keras.datasets  import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.utils import np_utils
from keras.regularizers import l2

batch_size = 128
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between train and test sets
(X_train, y_train),(X_valid, y_valid),(X_test, y_test) = mnist.load_data()
print X_train.shape, X_test.shape

X_train = X_train.reshape(50000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Dense(500, input_shape=(784,), W_regularizer=l2(0.0001)))
model.add(Activation('tanh'))
model.add(Dense(10, W_regularizer=l2(0.0001)))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
y_pred = model.predict(X_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print("first 10 pred numbers:", np.argmax(y_pred, axis = 1)[:10])
