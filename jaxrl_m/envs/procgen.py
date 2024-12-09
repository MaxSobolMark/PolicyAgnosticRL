import math

import jax
import numpy as np
from procgen import ProcgenEnv

from jaxrl_m.common.evaluation import parallel_evaluate
from jaxrl_m.data.dataset import Dataset


def get_procgen_dataset(buffer_fname):

    print(f"Attempting to load buffer: {buffer_fname}")
    buffer = np.load(buffer_fname)
    buffer = {k: buffer[k] for k in buffer}
    print(f'Loaded buffer with {len(buffer["action"])} observations')

    env_infos = dict(level_id=buffer["prev_level_seed"])
    print(buffer.keys())
    if "qstar" in buffer:
        env_infos = dict(
            level_id=buffer["prev_level_seed"],
            qstar=buffer["qstar"],
        )
        print("Using QSTAR")

    dataset_dict = dict(
        observations=np.moveaxis(buffer["observation"], 1, -1),
        actions=buffer["action"],
        rewards=buffer["reward"],
        masks=np.logical_not(buffer["done"]),
        next_observations=np.moveaxis(buffer["next_observation"], 1, -1),
        env_infos=env_infos,
    )
    return Dataset(dataset_dict)


def level_subset(dataset, size=10, level_ids=None):
    if level_ids is not None:
        valid_idxs, _ = np.nonzero(np.isin(dataset.env_infos["level_id"], level_ids))
    else:
        (valid_idxs,) = np.nonzero(dataset["env_infos"]["level_id"] < size)

    print(f"New dataset has size {len(valid_idxs)} / {dataset.size}")
    return Dataset(dataset.get_subset(valid_idxs))


class ProcgenWrappedEnv:
    def __init__(
        self, num_envs, env_name, start_level, num_levels, distribution_mode="easy"
    ):
        self.envs = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            distribution_mode=distribution_mode,
            start_level=start_level,
            num_levels=num_levels,
        )
        self.r = np.zeros(num_envs)
        self.t = np.zeros(num_envs)

    def obs(self, x):
        return x["rgb"]

    def reset(self):
        self.r = np.zeros_like(self.r)
        self.t = np.zeros_like(self.r)
        return self.obs(self.envs.reset())

    def step(self, a):
        obs, r, done, info = self.envs.step(a)
        self.r += r
        self.t += 1
        for n in range(len(done)):
            if done[n]:
                info[n]["episode"] = dict(
                    r=self.r[n], t=self.t[n], time_r=max(-500, -1 * self.t[n])
                )
                self.r[n] = 0
                self.t[n] = 0
        return self.obs(obs), r, done, info


def evaluate(
    policy_fn, env_name, start_level, num_levels, n_processes, num_eval, verbose=True
):
    envs = ProcgenWrappedEnv(n_processes, env_name, start_level, num_levels)
    returns, time_returns = parallel_evaluate(policy_fn, envs, num_eval, verbose)
    return {
        **{
            "mean return": np.mean(returns),
            "max return": np.max(returns),
            "median return": np.median(returns),
            "min return": np.min(returns),
        },
        **{
            "mean time return": np.mean(time_returns),
            "max time return": np.max(time_returns),
            "median time return": np.median(time_returns),
            "min time return": np.min(time_returns),
        },
    }
