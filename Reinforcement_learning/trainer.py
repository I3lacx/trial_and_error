import numpy as np
import matplotlib.pyplot as plt
from environments import create_environment


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