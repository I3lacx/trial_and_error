
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
from time import sleep

num_actions = 3  # [move_up, stay, move_down]


def defineModel():
    hidden_size = 12
    input_size = 6

    model = Sequential()
    # TODO: layers not correct
    model.add(Dense(hidden_size, input_shape=(input_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(optimizer="rmsprop", loss="mse")
    return model


def train(game, model):
    epsilon = 0.001
    epoch = 1000

    for e in range(epoch):
        game.newRound()
        game_over = False
        total_ball_speed = 0
        total_frames = 0
        bounce_count = 0

        # get initial input
        new_game_state = game.getCurrentState()
        input_arr = []
        target_arr = []

        while not game_over:
            old_game_state = new_game_state
            # get next action with epsilon as exploration
            if e < 10:
                action = np.random.randint(0, num_actions)
            elif np.random.rand() <= (epoch-e*30) * epsilon:
                # choose random action
                action = np.random.randint(0, num_actions)
            else:
                q = model.predict(old_game_state, batch_size=1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            bounce_count = game.getCurrentBounceCount()
            game_over = game.run_frame(action)
            new_game_state = game.getCurrentState()

            # reward = 1 if the y axis of the player and the ball align
            # reward should be if your action is towards the ball it is good, otherwise no

            total_ball_speed += np.abs(game.ball.speed[0])
            total_frames += 1

            # divided by 4 -> if the ball is in radius 1/4 of player height
            reward = get_reward_fast_ball(game, action)

            """
            # scales how much sharp the reward function scales
            reward_scaling = 10
            player_y = game.player1.pos_y
            ball_y = game.ball.pos_y
            dif_y = np.abs(player_y - ball_y) / pong.GAME_HEIGHT
            reward = (1-dif_y)**reward_scaling
            """

            """
            new_bounce_count = game.getCurrentBounceCount()
            if new_bounce_count > old_bounce_count:
                reward = 10
            elif game_over:
                reward = -10
            else:
                reward = 0
            """

            # store experience
            exp_replay = memory.ExperienceReplay()
            exp_replay.remember([old_game_state, new_game_state, reward, action], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model)

            # input to batch should be:
            # sequence of inputs where a full sequence contains one full game
            # depending on the result (+1 or -1) the path that has been taken
            # will be rewarded with -1 or +1 (* discount)
            # inputs shape: (1,6) -> ball position player x position etc.
            # targets. shape: (1,3) -> all possible actions

            # unnecessary complex solution
            # if np.max(targets) > 1 or np.min(targets) < -1:
            if list(input_arr):
                input_arr = np.vstack((input_arr, inputs))
                target_arr = np.vstack((target_arr, targets))
            else:
                input_arr = inputs
                target_arr = targets

        if list(input_arr):
            model.fit(input_arr, target_arr, batch_size=50, epochs=20, verbose=0)
            if e > 20:
                if input_arr.shape[0] == 1:
                    prediction = model.predict(input_arr)
                else:
                    x_arr = np.reshape(input_arr[input_arr.shape[0]-1], (1, 6))
                    prediction = model.predict(x_arr)
                # print("Target:", target_arr[input_arr.shape[0]-1])
                # print("Prediction:", prediction)

        print("Epoch {:03d} | Ball speed {:.4f} | Bounces {}"
              .format(e, total_ball_speed/total_frames, bounce_count))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)


def get_reward_fast_ball(game, action):
    player_y = game.player1.pos_y + game.player1.height / 2
    ball_y = game.ball.pos_y
    dif_y = player_y - ball_y

    stay_still_frame = game.player1.height / 4
    if (dif_y < 0 and action == 2) or (dif_y > 0 and action == 0) \
            or (np.abs(dif_y) < stay_still_frame and action == 1):
        reward = 1 * game.ball.speed[0] ** 2  # times x speed of ball, more reward

    else:
        reward = 0
    return reward


# No doesn't work aswell
def get_reward_slow_ball(game, action):
    player_y = game.player1.pos_y + game.player1.height / 2
    ball_y = game.ball.pos_y
    dif_y = player_y - ball_y

    stay_still_frame = game.player1.height / 4
    if (dif_y < 0 and action == 2) or (dif_y > 0 and action == 0) \
            or (np.abs(dif_y) < stay_still_frame and action == 1):
        reward = 1 * 4/(game.ball.speed[0] ** 2)  # times x speed of ball, more reward

    else:
        reward = 0
    return reward


# Not working as I want to
def get_reward_chillin(game, action):
    player_y = game.player1.pos_y + game.player1.height / 2
    ball_y = game.ball.pos_y
    dif_y = player_y - ball_y

    stay_still_frame = game.player1.height / 4
    if (dif_y < 0 and action == 2) or (dif_y > 0 and action == 0) \
            or (np.abs(dif_y) < stay_still_frame and action == 1):
        reward = 1  # times x speed of ball, more reward
    elif action == 1:
        reward = 0.3
    else:
        reward = 0
    return reward


def main():
    # init
    print("Initialize")
    model = defineModel()
    game = pong.initStandardGame(showGame=True, letAIPlay=True)
    train(game, model)


if __name__ == "__main__":
    main()
