import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

model = Sequential()
model.add(Dense(5, input_shape=(3**2,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(7))

model.save('my_model.h5')
del model

model = keras.models.load_model('my_model.h5')
