
## dependencies
import numpy as np
import math
import matplotlib.pyplot as plt

import os
import time
import sys
import copy
from typing import Optional, Union

import gym
from gym import error, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.utils import seeding



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class GridWorldEnv(gym.Env):
    """
    Grid World involves a 5 x 5 grid, with two target squares. The agent may move in any direction (right,
    left, up, down).

    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    The number of possible observations is dependent on the size of the map.
    For example, the 5x5 map has 25 possible observations.

    ### Rewards

    Reward schedule:
    - Start in target state 1 (G1): +10
    - Start in target state 2 (G2): +5
    - Start in any other state, valid move (M): 0
    - Move out of grid (O): -1



    ### Arguments

    ```
    gym.make('GridWorld-v0', target_state_1 = 3, target_state_2 = 7)
    ```

    `target_state_1`: Used to specify the [row,col] of the first target state

    `target_state_2`: Used to specify the [row,col] of the second target state

    `start_state`: Used to specify the [row,col] of the second target state



    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    def __init__(self, state = np.array([2,2]), target_state_1 = np.array([0,1]),
                 target_state_2 = np.array([0,3]), reset_state_1 = np.array([4,1]),
                 reset_state_2 = np.array([2,3]),nrow = 5, ncol = 5):

        # start location
        if state is None:
            state = np.array([2,2])

        self.state = state

        # target state locations
        if target_state_1 is None:
            target_state_1 = np.array([0,1])

        if target_state_2 is None:
            target_state_2 = np.array([0,4])

        self.target_state_1 = target_state_1
        self.target_state_2 = target_state_2

        # reset locations
        if reset_state_1 is None:
            reset_state_1 = np.array([4,1])

        if reset_state_2 is None:
            reset_state_2 = np.array([2, 3])

        self.reset_state_1 = reset_state_1
        self.reset_state_2 = reset_state_2

        # grid size
        if nrow is None:
            nrow = 5

        if ncol is None:
            ncol = 5

        self.nrow = nrow
        self.ncol = ncol
        self.shape = (self.nrow,self.ncol)

        # reward range
        #self.reward_range(-1,10)

        # action space
        self.actions = ['left','down','right','up']
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)

        # observation space
        self.n_states = self.nrow*self.ncol
        self.observation_space = spaces.Discrete(self.n_states)

        # initial state distribution
        self.initial_state_distrib = np.zeros(self.n_states)
        self.initial_state_distrib[self.state_to_ind(self.state)] = 1.0

        # Calculate transition probabilities and rewards
        self.probs = {}
        for s in range(self.n_states):
            position = np.unravel_index(s, self.shape)
            self.probs[s] = {a: [] for a in self.actions}
            self.probs[s]["up"] = self.compute_transition_probs(position, "up")
            self.probs[s]["right"] = self.compute_transition_probs(position, "right")
            self.probs[s]["down"] = self.compute_transition_probs(position, "down")
            self.probs[s]["left"] = self.compute_transition_probs(position, "left")


    # get observations
    def get_observation(self):
        obs = np.zeros(self.n_states)
        obs[self.state_to_ind(self.state)] = 1.0
        return obs

    # get actions
    def get_actions(self):
        return self.actions

    # compute new state
    def compute_new_state(self,state, action):
        row = state[0]
        col = state[1]

        # if in target state, reset
        if np.array_equal(state,self.target_state_1):
            row = self.reset_state_1[0]
            col = self.reset_state_1[1]

        if np.array_equal(state,self.target_state_2):
            row = self.reset_state_2[0]
            col = self.reset_state_2[1]

        # otherwise
        if action == "left":
            col = max(col - 1, 0)
        elif action == "down":
            row = min(row + 1, self.nrow - 1)
        elif action == "right":
            col = min(col + 1, self.ncol - 1)
        elif action == "up":
            row = max(row - 1, 0)

        new_state = np.array([row,col])

        return new_state

    # get reward
    def get_reward(self, state, action):
        new_state = self.compute_new_state(state, action)
        if np.array_equal(state, self.target_state_1):
            reward = +10
        elif np.array_equal(state, self.target_state_2):
            reward = +5
        elif np.array_equal(state, new_state):
            reward = -1
        else:
            reward = 0
        return reward


    # compute (prob, new_state, reward) - transition prob always 1
    def compute_transition_probs(self, state, action):
        new_state = self.compute_new_state(state, action)
        reward = self.get_reward(state,action)
        return [(1, new_state, reward)]

    # compute (new_state,reward,action,prob) - transition prob always 1
    def step(self,action):
        transitions = self.probs[self.state_to_ind(self.state)][action]
        ind = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, state, reward = transitions[ind]
        self.state = state
        self.last_action = action
        return (state, reward, action, {"prob": prob})

    # reset
    def reset(self,*,seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,):
        super().reset(seed=seed)
        state_ind = categorical_sample(self.initial_state_distrib, self.np_random)
        self.state = self.ind_to_state(state_ind)
        self.last_action = None

        if not return_info:
            return self.state
        else:
            return self.state, {"prob": 1}

    # render
    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        for ind in range(self.n_states):
            state = self.ind_to_state(ind)
            if np.array_equal(state,self.state):
                output = " x "
            # Print target states
            elif np.array_equal(state, self.target_state_1):
                output = " A "
            elif np.array_equal(state, self.target_state_2):
                output = " B "
            else:
                output = " o "

            if state[1] == 0:
                output = output.lstrip()
            if state[1] == self.ncol- 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()


    # convert coordinates to index
    def state_to_ind(self,state):
        row = state[0]; col = state[1]
        return row * self.ncol + col

    def ind_to_state(self,ind):
        return np.array([math.floor(ind / self.nrow),ind % self.nrow])

    def action_to_ind(self,action):
        if action == "left":
            return 0
        elif action == "down":
            return 1
        elif action == "right":
            return 2
        elif action == "up":
            return 3


class Agent:

    def __init__(self):
        self.total_reward = 0

    def step(self,env:GridWorldEnv):
        current_obs = env.get_observation()
        actions = env.get_actions()
        current_action = random.choice(actions)
        reward = env.take_action(current_action)
        total_reward += reward
