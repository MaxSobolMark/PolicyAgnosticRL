import d4rl
import gym
import numpy as np


class PointMaze:
    def __init__(self, maze_id="umaze"):
        self.env = gym.make(f"maze2d-{maze_id}-v1")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_obs(self):
        obs = self.env.unwrapped._get_obs()
        # Return only the position, not the velocity
        return obs[:2]

    def reset(self):
        obs = self.env.reset()
        return obs[:2]

    def step(self, action):
        # Set current velocity to be the same direction as the action with magnitude 2
        velocity = action / np.linalg.norm(action) * 2

        self.set_state(self.sim.data.qpos, velocity)

        obs, rew, done, info = self.env.step(action)
        obs = obs[:2]
        return obs, rew, done, info
