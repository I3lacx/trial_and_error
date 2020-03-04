""" File to import Bandit and other environments"""
import numpy as np


class Bandit:
    """ Single armed bandit to draw from normal, with mean and variance """
    def __init__(self, mean, variance=1):
        self.mean = mean
        self.variance = variance

    def draw(self):
        sample = np.random.normal(self.mean, self.variance)
        return sample


def create_environment():
    bandits = []
    for _ in range(21):
        bandits.append(Bandit(0))
    for _ in range(2):
        bandits.append(Bandit(1))
    for _ in range(1):
        bandits.append(Bandit(2))
    for _ in range(1):
        bandits.append(Bandit(3))

    return bandits
