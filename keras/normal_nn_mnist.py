import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.datasets import mnist
#import keras.utils.to_categorical as one_hot
from keras.utils import to_categorical as one_hot

from keras.callbacks import History

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

defined_models= [['categorical_crossentropy', 'rmsprop'], ['logcosh', 'sgd'],
                ['mean_squared_error','adagrad'], ['hinge','adam']]
model_scores = []

def nn_model(my_loss, my_optimizer):
    model = Sequential()
    model.add(Dense(32,input_shape=(28,28)))
    model.add(Activation('elu'))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(10,activation='softmax'))

    model.compile(loss= my_loss,
                  optimizer= my_optimizer,
                  metrics=['accuracy'])

    history = History()
    model.fit(X_train,Y_train, epochs=3, batch_size=128, callbacks=[history])
    score = model.evaluate(X_test, Y_test, batch_size=24)
    plt.subplot(111)
    plt.plot(history.history['acc'])
    model_scores.append([my_loss, my_optimizer, score])


# loss_models = ['categorical_crossentropy', 'logcosh', 'mean_squared_error', 'hinge']
# optimizer_models = ['rmsprop', 'sgd', 'adagrad', 'adam']


# for loss in loss_models:
#     for optimizer in optimizer_models:
#         nn_model(loss, optimizer)

good_loss = ['mean_squared_error', 'categorical_crossentropy']
good_optimizer = ['rmsprop','adam']

for loss in good_loss:
    for optimizer in good_optimizer:
        nn_model(loss, optimizer)

for elem in model_scores:
    print("Score of model {0}, {1}: accuracy: {2:.4f}, loss: {3:.4f} "\
                            .format(elem[0], elem[1], elem[2][1], elem[2][0]))

plt.show()

# keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,  \
#                             save_best_only=False, save_weights_only=False, mode='auto', period=1)
