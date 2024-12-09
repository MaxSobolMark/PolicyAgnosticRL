from gym import Env, RewardWrapper


class ScaledRewardWrapper(RewardWrapper):
    def __init__(self, env: Env, scale: float, bias: float):
        super().__init__(env)
        self.scale = scale
        self.bias = bias

    def reward(self, reward):
        return reward * self.scale + self.bias
