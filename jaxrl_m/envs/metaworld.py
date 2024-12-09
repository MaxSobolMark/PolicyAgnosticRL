from typing import Optional
import gym
import numpy as np
from dotmap import DotMap
from metaworld import MT1
from metaworld import policies


def get_scripted_policy(task_name):
    task_name = task_name.replace("-v2", "")
    # Add "Sawyer" at the beginning, make cammelcase, remove hyphens, and add "V2Policy" at the end
    policy_class_name = (
        "Sawyer"
        + "".join([word.capitalize() for word in task_name.split("-")])
        + "V2Policy"
    )
    policy = getattr(policies, policy_class_name)()

    # def sample_actions(obs, **kwargs):
    #     return policy.get_action(obs)

    # policy.sample_actions = sample_actions

    return policy


class MetaworldWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        seed,
        task_name,
        mt1: Optional[MT1] = None,
        min_task_index: int = 0,
        max_task_index: int = 50,
    ):
        super(MetaworldWrapper, self).__init__(env)
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.mt1 = mt1 or MT1(task_name, seed=seed)
        self.min_task_index = min_task_index
        self.max_task_index = max_task_index
        self.current_task_index = min_task_index

    def reset(self, *args, **kwargs):
        self.env.set_task(self.mt1.train_tasks[self.current_task_index])
        self.current_task_index = self.current_task_index + 1
        if self.current_task_index == self.max_task_index:
            self.current_task_index = self.min_task_index

        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        obs, rew, done, truncated, info = self.env.step(*args, **kwargs)
        rew = info["success"]
        done = done or info["success"] == 1.0 or truncated
        return obs, rew, done, truncated, {}

    def render(self, *args, **kwargs):
        return self.env.render()


def make_metaworld_env(environment_name, seed, **kwargs):
    task_name = environment_name[len("metaworld-") :]
    mt1 = MT1(task_name, seed=seed)
    e = mt1.train_classes[task_name](
        render_mode="rgb_array", camera_name="behind_gripper"
    )
    e.unwrapped.spec = DotMap(id=environment_name)
    e = MetaworldWrapper(e, seed, task_name, mt1=mt1, **kwargs)
    return e
