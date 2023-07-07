from abc import abstractmethod

import random

import numpy as np
from numpy import ndarray

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """
    π: ndarray
    v_values: ndarray
    q_values: ndarray

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        super().__init__(environment, λ, γ, t_max)
        # learning rate
        self.α = α

        # policy table
        self.π = np.full((self.env.n_actions, self.env.state_size), fill_value=1 / self.env.n_actions)
        # print(self.π)

        # state value table
        self.v_values = np.zeros((self.env.state_size,))
        # print(self.v_values)

        # state-action table
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))
        # print(self.q_values)

    @abstractmethod
    def learn(self, episode: Episode):
        # subclasses insert their implementation at this point
        # see for example be\kdg\rl\learning\tabular\qlearning.py
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_learning_start(self):
        self.t = 0

    def next_action(self, s: int):
        a = np.random.choice(range(self.env.n_actions), p=self.π[:, s])

        return a

    def evaluate(self):
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s])

        pass

    def improve(self):
        for s in range(self.env.state_size):
            # get the best action based on the q-value of the actions for this state
            # tie-breaking: if there are multiple actions with the same q-value, choose one of them randomly
            best_a = np.random.choice(np.where(self.q_values[s] == np.max(self.q_values[s]))[0])

            # if the action is the best action
            for a in range(self.env.n_actions):
                if a == best_a:
                    self.π[a, s] = 1 - self.ϵ + (self.ϵ / self.env.n_actions)
                    print(self.q_values)
                else:
                    self.π[a, s] = self.ϵ / self.env.n_actions

            self.decay()
        pass
