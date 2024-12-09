from collections import defaultdict
from typing import Optional

import numpy as np

from jaxrl_m.common.evaluation import add_to


class TrajSampler(object):
    """
    sampling trajectories for training or eval, with the option of adding the
    trajectories into the replay buffer.
    Apply reward and action modifications on the trajectories returned.
    """

    def __init__(
        self, env, clip_action, reward_scale, reward_bias, max_traj_length=1000
    ):
        self.clip_action = clip_action
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(
        self,
        policy_fn,
        num_episodes,
        replay_buffer=None,
        goal_relabel_fn: Optional[callable] = None,
        calc_mc_return_fn: Optional[callable] = None,
        store_max_trajectory_reward: bool = False,
        terminate_on_success: bool = False,
    ):
        """
        args:
            replay_buffer: if not None, insert the trajectories into the replay buffer
            goal_relabel_fn: if not None, relabel the goal of each transition and store in
                                transition["goals"].
                                This function should be f(trajectory) -> traj
                                the policy_fn should be f(observation, goal) -> action
            calc_mc_return_fn: if not None, calculate the MC return and add to each transition.
                                This function should be f(rewards, dones) -> mc_returns
            store_max_trajectory_reward: if True, store the max reward of the trajectory on the
                                replay buffer.
        """
        trajectories = []

        for _ in range(num_episodes):
            trajectory = defaultdict(list)
            reset_variables = self._env.reset()
            if isinstance(reset_variables, np.ndarray) or isinstance(
                reset_variables, dict
            ):
                observation = reset_variables
                info = {}
            else:
                assert len(reset_variables) == 2
                observation, info = reset_variables
            done = False
            step = 0
            while not done and step < self.max_traj_length:
                if goal_relabel_fn is not None:
                    action = policy_fn(observation, info["goal"])
                else:
                    action = policy_fn(observation)
                step_variables = self._env.step(action)
                if len(step_variables) == 5:
                    next_observation, r, terminated, truncated, info = step_variables
                    done = terminated  # or truncated
                else:
                    assert len(step_variables) == 4
                    next_observation, r, done, info = step_variables
                transition = dict(
                    observations=observation,
                    next_observations=next_observation,
                    actions=np.clip(action, -self.clip_action, self.clip_action),
                    rewards=r * self.reward_scale + self.reward_bias,
                    terminals=done,
                    masks=1.0 - done,
                )
                add_to(trajectory, transition)

                # terminate on success
                if (
                    terminate_on_success
                    and (r * self.reward_scale + self.reward_bias) == 0
                ):
                    break

                observation = next_observation
                step += 1

            # get MC return
            if calc_mc_return_fn is not None:
                mc_returns = calc_mc_return_fn(
                    trajectory["rewards"], trajectory["masks"]
                )
                trajectory["mc_returns"] = mc_returns

            # goal relabeling
            if goal_relabel_fn is not None:
                trajectory = goal_relabel_fn(trajectory)

            # insert into replay buffer
            if replay_buffer is not None:
                for i in range(len(trajectory["rewards"])):
                    transition = dict(
                        observations=trajectory["observations"][i],
                        actions=trajectory["actions"][i],
                        rewards=trajectory["rewards"][i],
                        next_observations=trajectory["next_observations"][i],
                        masks=trajectory["masks"][i],
                    )
                    if goal_relabel_fn is not None:
                        transition["goals"] = trajectory["goals"][i]
                    if calc_mc_return_fn is not None:
                        transition["mc_returns"] = trajectory["mc_returns"][i]
                    if store_max_trajectory_reward:
                        transition["max_trajectory_rewards"] = np.max(
                            trajectory["rewards"]
                        )

                    replay_buffer.insert(transition)

            trajectories.append(trajectory)

        return trajectories


def calc_return_to_go(rewards, masks, gamma, push_failed_to_min=True, min_reward=0.0):
    """default calc return_to_go function"""
    if rewards[-1] == min_reward and push_failed_to_min:
        # failed trajectory
        assert np.all(np.array(rewards) <= 0)
        reward_to_go = [min_reward / (1 - gamma)] * len(rewards)
    else:
        # success trajectory
        reward_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            reward_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * masks[-i - 1]
            prev_return = reward_to_go[-i - 1]

    return np.array(reward_to_go, dtype=np.float32)
