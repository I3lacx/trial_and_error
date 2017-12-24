import numpy as np
import matplotlib.pyplot as plt

import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

'''
This class representes a neural network so that its already usable
so you can use the NN to predict up or down
the first version is very basic but it will be improved upon
hard coded stuff:
layers: 3 (input, hidden1, hidden2, output)
input = ball_x cord, ball_y cord, , ball_x speed, ball_y speed, player_x cord, player_y cord
output = 1: prob to go up, 2 : porb to go down
size of layers: (6, 12, 6, 2)
activation = elu and last is softmax
'''

class Neural_network(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(16,input_shape=(6,1)))
        self.model.add(Activation('elu'))
        self.model.add(Dense(6))
        self.model.add(Activation('elu'))
        self.model.add(Flatten())
        self.model.add(Dense(2,activation='softmax'))

    def run(self, input):
        return self.model.predict(input, batch_size=1)

my_NN = Neural_network()
my_input = np.array([[[0.1,0.2,0.6,0.3,0.8,0.7]]])
my_input = my_input.reshape(1,6,1)
print(my_input)
print(my_NN.run(my_input))
