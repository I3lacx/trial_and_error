import sklearn.datasets
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#Set random seed for reproducibility
np.random.seed(42)

#one hot encoder
enc = OneHotEncoder()

#load the iris data Set
#iris_data is a dictonary with keys: data, target, target_names
iris_data = sklearn.datasets.load_iris()

#convert the "traget"(labels of the data) to one hot labels
labels = iris_data['target']
labels = labels.reshape(-1,1)
enc.fit(labels)
print("Number of different labels: ",enc.n_values_[0])
one_hot_labels = enc.transform(labels).toarray()
#print(one_hot_labels)

#define X and Y
X = iris_data['data'] #shape: (150,4)
Y = one_hot_labels    #shape: (150,3)

#model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=150, batch_size=10)
scores = model.evaluate(X,Y)
print("Names:{} with scores: {}%".format(model.metrics_names[1], np.round(scores[1]*100,2)))
