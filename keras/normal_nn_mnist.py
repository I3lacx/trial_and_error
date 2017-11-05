import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.datasets import mnist
#import keras.utils.to_categorical as one_hot
from keras.utils import to_categorical as one_hot

np.random.seed(42)

train, test = mnist.load_data()
X_train = train[0]
Y_train = train[1]
X_test = test[0]
Y_test = test[1]

#convert and normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = one_hot(Y_train, 10)
Y_test = one_hot(Y_test, 10)

model = Sequential()
model.add(Dense(32,input_shape=(28,28)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=10, batch_size=12)
score = model.evaluate(X_test, Y_test, batch_size=12)
print(score)
