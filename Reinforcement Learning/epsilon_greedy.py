import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    """ Single armed bandit to draw from normal, with mean and variance """
    def __init__(self, mean, variance=1):
        self.mean = mean
        self.variance = variance

    def draw(self):
        sample = np.random.normal(self.mean, self.variance)
        return sample


class EpsilonGreedy:
    """ Epsilon Greedy algorithm """
    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.bandit_means = np.zeros(num_actions)
        self.number_draws = np.zeros(num_actions)
        self.rewards = []
        self.total_reward = 0

    def act(self, bandits):
        epsilon_check = np.random.random()

        if epsilon_check < self.epsilon:
            action = np.random.randint(0, len(bandits))
        else:
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


def create_environment():
    bandits = []
    for _ in range(5):
        bandits.append(Bandit(0))
    for _ in range(2):
        bandits.append(Bandit(1))
    for _ in range(1):
        bandits.append(Bandit(2))
    for _ in range(1):
        bandits.append(Bandit(3))

    return bandits


def create_agents(bandits):
    agents = []
    num_actions = len(bandits)
    for i in np.arange(0,1,0.1):
        agents.append(EpsilonGreedy(round(i,3), num_actions))
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
        print(f"Agent {i} with epsilon {agent.epsilon} got reward:", agent.total_reward)
        plt.plot(cum_reward, label=f'eps = {agent.epsilon}')

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
