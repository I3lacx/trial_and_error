"""
Idea, you know that the Mean of the Bandits is much lower than e.g. 10
So if you choose 10, then you will have to explore the other ones if you
set this as initial value.
So no epsilon random thing, and set high initial means

"""
import numpy as np
import matplotlib.pyplot as plt
from environments import create_environment


class OptimisticInitialValue:
    """ Optimistic Initial Value algorithm """
    def __init__(self, high_value, num_actions):
        self.high_value = high_value
        self.bandit_means = np.ones(num_actions) * high_value
        self.number_draws = np.zeros(num_actions)
        self.rewards = []
        self.total_reward = 0

    def act(self, bandits):
        action = np.argmax(self.bandit_means)

        drawn_sample = bandits[action].draw()
        self.number_draws[action] += 1
        self.update(action, drawn_sample)

    def update(self, action, sample):
        old_mean = self.bandit_means[action]
        n = self.number_draws[action]
        self.bandit_means[action] = (1-1/n) * old_mean + 1/n * sample
        self.total_reward += sample
        self.rewards.append(sample)


def create_agents(bandits):
    agents = []
    num_actions = len(bandits)
    for i in np.arange(1, 30, 5):
        agents.append(OptimisticInitialValue(i, num_actions))
    return agents


def train(agent, bandits, epochs):
    """ Train the agent """
    for i in range(epochs):
        agent.act(bandits)

    cummulative_reward = np.cumsum(agent.rewards) / np.arange(1, epochs+1)
    return cummulative_reward


def main():
    bandits = create_environment()
    agents = create_agents(bandits)
    epochs = 1000

    # Train and plot every bandit
    for i, agent in enumerate(agents):
        cum_reward = train(agent, bandits, epochs)
        print(f"Agent {i} with mean {agent.high_value} got reward:", agent.total_reward)
        plt.plot(cum_reward, label=f'eps = {agent.high_value}')

    # Plot bandits
    for bandit in bandits:
        plt.plot(np.ones(epochs) * bandit.mean)

    plt.legend()
    # plt.xscale('log')
    plt.show()


if __name__ ==  "__main__":
    print("--------START---------")
    main()
    print("---------END----------")