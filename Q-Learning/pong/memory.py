import numpy as np


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, gameOver):
        """
        @inputs:
        states -> array containing all information over the current game state
        states[0] -> oldGameState
        states[1] -> newGameState
        states[2] -> reward
        states[3] -> action taken
        """
        # TODO: clean up states so it isn't unecesarry high dimensional
        self.memory.append(states)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]

        gameState_dim = self.memory[0][0].shape[1]

        # in each cell one input is saved
        # TODO: 6 should represent the number of inputs
        inputs = np.zeros((len_memory, 6))
        # in each cell one output array so for each action one reward
        # so like one hot coding of action?
        # TODO: the number represents the number of actions possible
        targets = np.zeros((len_memory, 3))

        for i, save in enumerate(self.memory):

            # state_t, action_t, reward_t, state_tp1 = self.memory[idx]

            # TODO: dictionary would be more pretty
            # inputs[i] old state at time i
            inputs[i] = save[0][0]

            # TODO: why [0] at the end? what is this x?
            # save prediction with future reward there (currently 0)
            # targets[i] = [[x,0] for x in model.predict(np.reshape(save[0][0],(6,1)).T)[0]]
            # why is my target what the model would have predicted anyway?
            targets[i] = model.predict(np.reshape(save[0][0], (6, 1)).T)[0]

            # maximum future reward
            max_future_reward = np.max(model.predict(np.reshape(save[1][0], (6, 1)).T)[0])

            # update chosen reward as not 0
            # save[2] -> reward
            # TODO: discount to the power of t?
            # target of action taken (save[3] -> action taken)
            targets[i][save[3]] = save[2] + self.discount * max_future_reward

        # returns input with corresponding reward
        return inputs, targets
