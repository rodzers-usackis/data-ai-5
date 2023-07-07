from abc import ABC
from be.kdg.rl.environment import markovdecisionprocess

import gym as gym
from gym.wrappers import TimeLimit

import pygame as pygame

from be.kdg.rl.environment.environment import Environment


class OpenAIGym(Environment, ABC):
    """
    Superclass for all kinds of OpenAI environments
    Wrapper for all OpenAI Environments
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._env: TimeLimit = gym.make(name, render_mode='ansi', is_slippery=True)
        self.reset()
        self.render()
        """
        running = True
        terminated = None
        truncated = None

        - 0: LEFT
        - 1: DOWN
        - 2: RIGHT
        - 3: UP
        

        while running:

            key_inputs = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if key_inputs[pygame.K_LEFT]:
                terminated, truncated = self.perform_user_action(0)
            if key_inputs[pygame.K_DOWN]:
                terminated, truncated = self.perform_user_action(1)
            if key_inputs[pygame.K_RIGHT]:
                terminated, truncated = self.perform_user_action(2)
            if key_inputs[pygame.K_UP]:
                terminated, truncated = self.perform_user_action(3)

            if terminated or truncated:
                self._env.reset()
                terminated = None
                truncated = None

            pygame.time.Clock().tick(10)

        self._env.close()
        """

    def perform_user_action(self, action: int):
        observation, reward, terminated, truncated, info = self.step(action)
        print(observation, reward, terminated, truncated, info)
        return terminated, truncated

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def state_size(self):
        if self.isdiscrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def isdiscrete(self) -> bool:
        return hasattr(self._env.observation_space, 'n')

    @property
    def name(self) -> str:
        return self._name


class FrozenLakeEnvironment(OpenAIGym):

    def __init__(self) -> None:
        super().__init__(name="FrozenLake-v1")


class CartPoleEnvironment(OpenAIGym):

    def __init__(self) -> None:
        super().__init__(name="CartPole-v0")
