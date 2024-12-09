"""got from Young's JaxCQL repo and Mitsuhiko's Cal-QL repo"""

import collections
from typing import Optional

from jaxrl_m.data.dataset import Dataset

try:
    import d4rl
except Exception:
    d4rl = None
    print("d4rl not installed, please install it if you want to use d4rl environments.")
import gym
import numpy as np
from gym import Wrapper
from jaxrl_m.envs.wrappers.video_recorder import VideoRecorder
from jaxrl_m.utils.train_utils import concatenate_batches

D4RL_ENV_CONFIG = {
    "antmaze": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
}


def make_d4rl_env(
    env_name: str,
    max_episode_steps: int,
    save_video: bool,
    save_video_dir: Optional[str] = None,
    save_video_prefix: Optional[str] = None,
    seed: int = 0,
    no_info_dict_on_reset: bool = False,
):
    try:
        env = gym.make(env_name, seed=seed)
    except TypeError:
        # some envs don't take in seed as argument
        env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = TruncationWrapper(env, no_info_dict_on_reset=no_info_dict_on_reset)

    if save_video:
        assert save_video_dir is not None
        assert save_video_prefix is not None
        env = VideoRecorder(
            env,
            save_folder=save_video_dir,
            save_prefix=save_video_prefix,
        )

    return env


class TruncationWrapper(Wrapper):
    """d4rl only supports the old gym API, where env.step returns a 4-tuple without
    the truncated signal. Here we explicity expose the truncated signal."""

    def __init__(self, env, no_info_dict_on_reset=False):
        super().__init__(env)
        self._no_info_dict_on_reset = no_info_dict_on_reset

    def reset(self):
        s = self.env.reset()
        if self._no_info_dict_on_reset:
            return s
        return s, {}

    def step(self, a):
        s, r, done, info = self.env.step(a)
        truncated = info.get("TimeLimit.truncated", False)
        return s, r, done, truncated, info


def get_d4rl_dataset(
    env,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
):
    dataset = d4rl.qlearning_dataset(env.unwrapped)

    if "kitchen" in env.spec.name:
        dataset["terminals"] = (dataset["rewards"] == 4).astype(np.float32)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset["rewards"])

    for i in range(len(dones_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dones_float,
        masks=1 - dataset["terminals"].astype(np.float32),
    )


def get_d4rl_dataset_with_mc_calculation(
    env_name,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset_path: Optional[str] = None,
):
    if "antmaze" in env_name or "maze2d" in env_name:
        is_sparse_reward = True
    elif "halfcheetah" in env_name or "hopper" in env_name or "walker" in env_name:
        is_sparse_reward = False
    elif "kitchen" in env_name:
        is_sparse_reward = False
    else:
        raise NotImplementedError

    if dataset_path is not None and dataset_path != "":
        dataset = np.load(dataset_path)
    else:
        dataset = None

    dataset = qlearning_dataset_and_calc_mc(
        gym.make(env_name).unwrapped,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
        dataset=dataset,
        is_sparse_reward=is_sparse_reward,
    )

    return Dataset(
        dict(
            observations=dataset["observations"],
            actions=dataset["actions"],
            next_observations=dataset["next_observations"],
            rewards=dataset["rewards"],
            mc_returns=dataset["mc_returns"],
            masks=1 - dataset["terminals"].astype(np.float32),
        )
    )


def qlearning_dataset_and_calc_mc(
    env,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    dataset=None,
    terminate_on_end=True,
    is_sparse_reward=True,
    **kwargs
):
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)
    # Recalculate timeouts
    timeouts = np.zeros_like(dataset["terminals"])
    if "kitchen" in env.spec.name:
        dataset["terminals"] = (dataset["rewards"] == 4).astype(np.float32)
    for i in range(len(dataset["terminals"]) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            timeouts[i] = True
    timeouts[-1] = True
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # iterate over transitions, put them into trajectories
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])

        is_final_timestep = timeouts[i]

        if (not terminate_on_end) and is_final_timestep or i == N - 1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in (
                    "actions",
                    "next_observations",
                    "observations",
                    "rewards",
                    "terminals",
                ):
                    data_[k].append(dataset[k][i])
            if "next_observations" not in dataset.keys():
                data_["next_observations"].append(dataset["observations"][i + 1])
            episode_step += 1

        if (done_bool or is_final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = (
                episode_data["rewards"] * reward_scale + reward_bias
            )
            episode_data["mc_returns"] = calc_return_to_go(
                env.spec.name,
                episode_data["rewards"],
                1 - episode_data["terminals"],
                gamma,
                reward_scale,
                reward_bias,
                is_sparse_reward,
            )
            episode_data["actions"] = np.clip(
                episode_data["actions"], -clip_action, clip_action
            )
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    return concatenate_batches(episodes_dict_list)


def calc_return_to_go(
    env_name, rewards, masks, gamma, reward_scale, reward_bias, is_sparse_reward
):
    """
    A config dict for getting the default high/low rewrd values for each envs
    """
    if len(rewards) == 0:
        return np.array([])

    if "antmaze" in env_name:
        reward_neg = (
            D4RL_ENV_CONFIG["antmaze"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        assert (
            not is_sparse_reward
        ), "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [reward_neg / (1 - gamma)] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                masks[-i - 1]
            )
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)
