import numpy as np
import models
import copy
"""
own implementation of Expectation Maximization
a optimizing algorithm usefull for incomplete data

Given a dataset complete or incomplete and given a structure
of a network? with/ without strutcure
Update weights such that model can fit the data perfectly
Data:
5 Features with values 0 or 1 and None for missing data point
"""


# Todo auto generate dataset from distributon/model then learn model

# complete dataset
data_1 = [[0, 1, 0, 1, 0],
          [0, 1, 1, 0, 1],
          [1, 0, 1, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 1, 0],
          [0, 1, 1, 0, 1]]

"""
# model consists of 5 independent bernoullis
# initialized with 0
model_1 = models.SimpleBernoulli(5)
print(model_1.prob(data_1))
print(model_1.log_prob(data_1))

# in the case of complete data we can count to calculate the best result:]
feature_count = np.zeros(5)
for sample_idx, sample in enumerate(data_1):
    for feature_idx, feature in enumerate(sample):
        feature_count[feature_idx] += feature

model_1.features = np.array([count/len(data_1) for count in feature_count])
print(model_1.features)
print(model_1.prob(data_1))
"""

# This should not be changed
data = [[0, 0, 1],
        [0, 1, None],
        [1, 0, 0],
        [1, None, 1],
        [0, 1, 0],
        [1, None, 0],
        [1, 0, 1],
        [None, 0, 1],
        [0, 1, None],
        [None, 0, 1]]


def get_predicted_data(model, data, hard_update=False):
    predicted_data = copy.deepcopy(data)
    for sample_idx, sample in enumerate(data):
        for var_idx, var in enumerate(sample):
            if var is None:
                if var_idx == 0:
                    if not hard_update:
                        b = data[sample_idx][1]
                        c = data[sample_idx][2]
                        predicted_data[sample_idx][var_idx] = model.p_a(b, c)
                    if hard_update:
                        predicted_data[sample_idx][var_idx] = np.rint(model.p_a())
                elif var_idx == 1:
                    a = data[sample_idx][0]
                    if not hard_update:
                        predicted_data[sample_idx][var_idx] = model.p_b_a(a)
                    if hard_update:
                        predicted_data[sample_idx][var_idx] = np.rint(model.p_b_a(a))
                elif var_idx == 2:
                    a = data[sample_idx][0]
                    if not hard_update:
                        predicted_data[sample_idx][var_idx] = model.p_c_a(a)
                    if hard_update:
                        predicted_data[sample_idx][var_idx] = np.rint(model.p_c_a(a))
    return predicted_data


def update_parameters(predicted_data):
    # by counting get new parameters:
    num_samples = len(predicted_data)
    # AB / AC
    # 00 01 10 11
    counts = [[0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

    for sample_idx, sample in enumerate(predicted_data):
        for feature_idx, feature_val in enumerate(sample):
            if feature_idx == 0:
                if feature_val == 0:
                    counts[feature_idx][0] += 1
                else:
                    counts[feature_idx][1] += 1
            elif feature_idx in [1, 2]:
                if sample[0] == 0 and feature_val == 0:
                    counts[feature_idx][0] += 1
                elif sample[0] == 0 and feature_val == 1:
                    counts[feature_idx][1] += 1
                elif sample[0] == 1 and feature_val == 0:
                    counts[feature_idx][2] += 1
                else:
                    counts[feature_idx][3] += 1
            else:
                raise ValueError("Feature index should be between 0-2 inc. but is ", feature_idx)
    new_params = {
        "A": counts[0][1] / np.sum(counts[0]),
        "B": [counts[1][1] / (counts[1][1] + counts[1][0]), counts[1][3] / (counts[1][2] + counts[1][3])],
        "C": [counts[2][1] / (counts[2][1] + counts[2][0]), counts[2][3] / (counts[2][2] + counts[2][3])]
    }
    model = models.BayesNet1(new_params)
    return model


def test_model(model, data):
    """
    :param model:
    :param data:
    :return:
    """
    # will be multiplied by the prob of each sample
    total_prob = 1
    for sample in data:
        total_prob *= model.prob(sample)
    print(total_prob)


# first param when A=0, second when A=1
init_params = {
    "A": 0.5,
    "B": [0.5, 0],
    "C": [0.5, 0.5]
}

random_init = {
    "A": np.random.rand(),
    "B": np.random.random(2),
    "C": np.random.random(2)
}

# given a factorization of the network I can calculate the probabilitios
# so each probability should have a function ish to be called so the network needs
# to know how to calcluate each variable
# also find the independencies?
# for continuos variables???
net = models.BayesNet1(init_params)
new_model = net
for i in range(3):
    print(new_model.params)
    new_data = get_predicted_data(new_model, data)
    # print(new_data)
    new_model = update_parameters(new_data)


