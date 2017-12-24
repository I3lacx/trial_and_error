import numpy as np
import matplotlib.pyplot as plt

import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

'''
from keras import backend as K
K.set_image_dim_ordering('th')
because the order of the images is differnet:
theano: NCHW, tensorflow: NHWC
Number, channels, height, width
with th it expects theano ordering
normaly it would expect tensorflow ordering
'''

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train: shape", X_train.shape)

plt.imshow(X_train[0])

#Reshape datasets
X_train = X_train.reshape(X_train.shape[0],28, 28,1) #1 for 1 depth
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train2 = keras.utils.normalize(X_train, order=10)
X_train /= 255

#print("X TRAIN:", X_train[0][0][0][0])
# for i in range(27):
#     if(X_train[0][0][12][i] != X_train2[0][0][12][i]):
#         print("Different Values for {0:.4f} and {1:.4f}".format(X_train[0][0][12][i], X_train2[0][0][12][i]))

X_test /= 255

#print(X_train[0])

y_train = np_utils.to_categorical(y_train, 10) #convert to binary one hot
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
#input_shape <=> 1 sample
#32 3 3 -> layers
model.add(Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print(score)
