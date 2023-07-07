from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning

import matplotlib.pyplot as plt
import numpy as np
import random

from be.kdg.rl.util.visualization import draw_quiver

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment()
    environment.reset()

    # create an Agent that uses Qlearning Strategy
    # agent: Agent = TabularAgent(environment, Qlearning(environment))
    # agent.train()

    # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(environment, NStepQlearning(environment, 5))
    # agent.train()

    # create an Agent that uses MonteCarloLearning Strategy
    agent: Agent = TabularAgent(environment, MonteCarloLearning(environment))
    agent.train()

    draw_quiver(agent, True, 'quiver.png', 'frozen_lake.png')
