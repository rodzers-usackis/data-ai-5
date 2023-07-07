import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def learn(self, episode: Episode):
        # get last percept from episode
        p = episode.percepts[-1]
        # update rule
        self.q_values[p.state][p.action] = self.q_values[p.state][p.action] + self.α * (p.reward + self.γ * np.max(self.q_values[p.next_state]) - self.q_values[p.state][p.action])

        super().learn(episode)


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # Check if we have enough percepts in the episode
        # if |E|≥N
        if episode.size >= self.n:
            # Loop through percepts in episode
            for p in episode.percepts:
                # Update rule
                # q(s,a) ←q(s,a) −α *(q(s,a) −[r(s,a) + γ ·maxa′(q(s′,a′))])
                self.q_values[p.state][p.action] = self.q_values[p.state][p.action] - self.α * (self.q_values[p.state][p.action] - (p.reward + self.γ * np.max(self.q_values[p.next_state])))

        super().learn(episode)


class MonteCarloLearning(NStepQlearning):
    # Monte Carlo is n step Q-learning with n = the length of the entire episode
    # We can use the NStepQlearning class as a base class
    # We only need to override the constructor

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        NStepQlearning.__init__(self, environment, environment.state_size, α, λ, γ, t_max)

    pass
