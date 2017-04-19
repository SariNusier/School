# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from pacman import Directions
from game import Agent
import random


class StateAsKey:
    """ 
    Helper class. Used as a key in a dictionary.
    The hashing and comparison functions are overriden in order to use the objects as keys in a dict.
    Help from:
    https://stackoverflow.com/questions/4901815/object-of-custom-type-as-dictionary-key
    """

    def __init__(self, state, action):
        self.pacpos = state.getPacmanPosition()
        self.ghostpos = frozenset(state.getGhostPositions())
        self.food = str(state.getFood())
        self.action = action

    def __hash__(self):
        return hash((self.pacpos, self.ghostpos, self.food, self.action))

    def __eq__(self, other):
        return (self.pacpos, self.ghostpos, self.food, self.action) == (
            other.pacpos, other.ghostpos, other.food, other.action)

    def __ne__(self, other):
        return not (self == other)


class QLearner:
    """
    This is the learner class.
    The methods' description are written below.
    """
    def __init__(self, alpha, epsilon, gamma):
        self.state_to_update_key = None
        self.state_to_update = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = {}

    def decide(self, state, is_final=False):
        """
        Returns an action and updates q-values. 
        For simplicity, the same function is called for terminal states as well, 
        with the by setting the is_final flag to True 
        """
        if self.state_to_update_key is not None:
            self.q_values[self.state_to_update_key] = self.compute_bellman(state)

        if not is_final:
            action = self.get_action(state)
            self.state_to_update_key = StateAsKey(state, action)
            self.state_to_update = state
            if not (self.state_to_update_key in self.q_values):
                self.q_values[self.state_to_update_key] = 0
            return action

        self.state_to_update_key = None
        self.q_values[StateAsKey(state, None)] = state.getScore()
        # We return None when the state is terminal (no action is needed)
        return None

    def compute_bellman(self, new_state):
        """
        Computes Q(s,a) when moving from s -> s'
        :param new_state: This is s' 
        :return: The new value of Q(s,a) 
        """
        prev_state = self.state_to_update
        # cur_q represents the current Q value of the state to be updated (prev_state): Q(s,a)
        cur_q = self.q_values[self.state_to_update_key]
        update_val = self.alpha * (prev_state.getScore() + self.gamma * self.get_max_Q(new_state)[1] - cur_q)
        return cur_q + update_val

    def get_action(self, state):
        """
        Chooses an action using epsilon greedy.
        :param state: State containing all the valid moves 
        :return: Returns a valid move for the state 
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if random.random() > self.epsilon:
            mx_q = self.get_max_Q(state)[0]
            return mx_q
        return random.choice(legal)

    def get_max_Q(self, state):
        """
        Finds the maximum Q value for the state and all the valid actions in that state.
        It returns both the action(argmax) and the maximum value.
        :param state: State we are interested in.
        :return: Returns argmax and the maximum value
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # When the state is terminal, there are no legal moves.
        if len(legal) == 0:
            return None, state.getScore()

        for action in legal:
            if not (StateAsKey(state, action) in self.q_values):
                self.q_values[StateAsKey(state, action)] = 0

        all_vals = [self.q_values[StateAsKey(state, action)] for action in legal]
        max_val = max(all_vals)
        arg_max = legal[all_vals.index(max_val)]

        return arg_max, max_val


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.3, epsilon=0.2, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.q_learner = QLearner(self.alpha, self.epsilon, self.gamma)

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        return self.q_learner.decide(state)

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print("A game just ended!")

        self.q_learner.decide(state, is_final=True)
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        print(self.getEpisodesSoFar())
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
            self.q_learner.alpha = 0
            self.q_learner.epsilon = 0
