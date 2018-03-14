
'''
Main class for pong game with q-learning

main stuff here
'''

__version__ = '0.1'
__author__ = 'I3lacx'

import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import copy
import pong
import memory

num_actions = 3  # [move_left, stay, move_right]

def defineModel():
	hidden_size = 4
	input_size = 6

	model = Sequential()
	#TODO: layers not correct
	model.add(Dense(hidden_size, input_shape=(input_size,), activation='relu'))
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dense(num_actions))
	model.compile(sgd(lr=.2), "mse")
	return model

def train(game, model):
	epsilon = 0.1
	epoch = 1000

	for e in range(epoch):
		loss = 0.
		game.newRound()
		gameOver = False
		winCount = 0
		reward = 0
		# get initial input
		newGameState = game.getCurrentState()

		while not gameOver:
			oldGameState = newGameState
			# get next action with epsilon as exploration
			if np.random.rand() <= epsilon:
				#choose random action
				action = np.random.randint(0, num_actions, size=1)
			else:
				q = model.predict(oldGameState, batch_size=1)
				action = np.argmax(q[0])

			# apply action, get rewards and new state
			gameOver = game.runFrame(action)
			newGameState = game.getCurrentState()
			if(game.getCurrentBounceCount() != reward):
				reward = game.getCurrentBounceCount()
				if(reward != 0):
					winCount += 1

			# store experience
			exp_replay = memory.ExperienceReplay()
			exp_replay.remember([oldGameState, newGameState, reward, action], gameOver)

			# adapt model
			inputs, targets = exp_replay.get_batch(model)

			loss += model.train_on_batch(inputs, targets)

		print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, winCount))

	# Save trained model weights and architecture, this will be used by the visualization code
	model.save_weights("model.h5", overwrite=True)
	with open("model.json", "w") as outfile:
		json.dump(model.to_json(), outfile)

def main():
	#init
	print("Initialize")
	model = defineModel()
	game = pong.initStandardGame(showGame=True,letAIPlay=True)
	train(game, model)

if __name__ == "__main__":
	main()
