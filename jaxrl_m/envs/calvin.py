from typing import Optional, Tuple
import copy
from collections import defaultdict
from contextlib import contextmanager

import cv2
import pickle
import gzip
import hydra
import numpy as np
import gym
from gym import spaces

from calvin_env.envs.play_table_env import PlayTableSimEnv
from collections import deque
from jaxrl_m.data.dataset import Dataset
from jaxrl_m.data.image_replay_buffer import ImageReplayBuffer
from jaxrl_m.data.bridge_dataset import glob_to_path_list


def get_dataset(
    dataset,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    filter_terminals=False,
    obs_dtype=np.float32,
):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dataset["terminals"][-1] = 1
    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset["terminals"])[0]
        last_idx = np.nonzero(dataset["terminals"])[0]
        penult_idx = last_idx - 1
        new_dataset = dict()
        for k, v in dataset.items():
            if k == "terminals":
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    dones_float = dataset["terminals"].copy()

    observations = dataset["observations"].astype(obs_dtype)
    next_observations = dataset["next_observations"].astype(obs_dtype)

    return Dataset(
        {
            "observations": observations,
            "actions": dataset["actions"].astype(np.float32),
            "rewards": dataset["rewards"].astype(np.float32),
            "masks": 1.0 - dones_float.astype(np.float32),
            "dones_float": dones_float.astype(np.float32),
            "next_observations": next_observations,
        }
    )


def get_calvin_config():
    hydra.initialize(config_path="calvin_config")
    cfg = hydra.compose(config_name="calvin")
    return cfg


def get_calvin_env(cfg=None, goal_conditioned: bool = False, **kwargs):
    if cfg is None:
        cfg = get_calvin_config()
    env = CalvinEnv(**cfg)
    env.max_episode_steps = cfg.max_episode_steps = 360
    env = GymWrapper(
        env=env,
        from_pixels=cfg.pixel_ob,
        from_state=cfg.state_ob,
        height=cfg.screen_size[0],
        width=cfg.screen_size[1],
        channels_first=False,
        frame_skip=cfg.action_repeat,
        return_state=False,
    )
    env = wrap_env(env, cfg)
    if goal_conditioned:
        env = GCCalvinWrapper(env, goal_image_size=cfg.screen_size[0], **kwargs)
    else:
        env = AddProprioWrapper(env)
    return env


def get_calvin_tfrecord_dataset(
    tfrecord_regexp: str,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    cache: bool = False,
    train: bool = True,
    seed: int = 0,
    **kwargs,
) -> ImageReplayBuffer:
    assert tfrecord_regexp.endswith("?*.tfrecord")
    paths = glob_to_path_list(tfrecord_regexp)
    return ImageReplayBuffer(
        data_paths=paths,
        seed=seed,
        goal_relabeling_strategy=goal_relabeling_strategy,
        goal_relabeling_kwargs=goal_relabeling_kwargs,
        cache=cache,
        train=train,
        **kwargs,
    )


def get_calvin_env_and_dataset(dataset_path: str):
    assert dataset_path.endswith("calvin.gz")
    env = get_calvin_env(goal_sampler=np.zeros((1, 1, 1, 1)))

    data = pickle.load(gzip.open(dataset_path, "rb"))
    ds = []
    for i, d in enumerate(data):
        if len(d["obs"]) < len(d["dones"]):
            continue  # Skip incomplete trajectories.
        # Only use the first 21 states of non-floating objects.
        d["obs"] = d["obs"][:, :21]
        new_d = dict(
            observations=d["obs"][:-1],
            next_observations=d["obs"][1:],
            actions=d["actions"][:-1],
        )
        num_steps = new_d["observations"].shape[0]
        new_d["rewards"] = np.zeros(num_steps)
        new_d["terminals"] = np.zeros(num_steps, dtype=bool)
        new_d["terminals"][-1] = True
        ds.append(new_d)
    dataset = dict()
    for key in ds[0].keys():
        dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
    dataset = get_dataset(dataset=dataset)

    return env, dataset


class CalvinEnv(PlayTableSimEnv):
    spec = gym.envs.registration.EnvSpec("calvin", "calvin")

    def __init__(self, tasks: dict = {}, **kwargs):
        self.max_episode_steps = kwargs.pop("max_episode_steps")
        self.reward_norm = kwargs.pop("reward_norm")
        self.markovian_rewards = kwargs.pop("markovian_rewards", True)
        self.include_distractors_in_state_obs = kwargs.pop(
            "include_distractors_in_state_obs", False
        )
        # remove unwanted arguments from the superclass
        [
            kwargs.pop(key)
            for key in [
                "id",
                "screen_size",
                "action_repeat",
                "frame_stack",
                "absorbing_state",
                "pixel_ob",
                "state_ob",
                "num_sequences",
                "data_path",
                "save_dir",
                "record",
            ]
        ]
        super().__init__(**kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=((21 if not self.include_distractors_in_state_obs else 39),),
        )

        self.tasks = hydra.utils.instantiate(tasks)
        self.target_tasks = list(self.tasks.tasks.keys())
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        self._t = 0
        self.sequential = False

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        self._t = 0
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        return obs

    def reset_to_state(self, robot_obs, scene_obs):
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.get_info()
        self._t = 0
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        return obs

    def get_obs(self):
        obs = self.get_state_obs()
        obs = np.concatenate([obs["robot_obs"], obs["scene_obs"]])
        if not self.include_distractors_in_state_obs:
            obs = obs[:21]
        return obs

    def _reward(self):
        current_info = self.get_info()
        completed_tasks = self.tasks.get_task_info_for_set(
            self.start_info, current_info, self.target_tasks
        )
        next_task = self.tasks_to_complete[0]

        reward = 0
        for task in list(completed_tasks):
            if self.sequential:
                if task == next_task:
                    reward += 1
                    self.tasks_to_complete.pop(0)
                    self.completed_tasks.append(task)
            else:
                if task in self.tasks_to_complete:
                    reward += 1
                    self.tasks_to_complete.remove(task)
                    self.completed_tasks.append(task)

        if self.markovian_rewards:
            reward = len(completed_tasks)
        reward *= self.reward_norm
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has completed all tasks. Should be called after _reward()."""
        done = len(self.tasks_to_complete) == 0
        d_info = {"success": done}
        return done, d_info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        for task in self.target_tasks:
            self.solved_subtasks[task] = (
                1 if task in self.completed_tasks or self.solved_subtasks[task] else 0
            )
        return info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        self._t += 1
        if self._t >= self.max_episode_steps:
            done = True
        return obs, reward, done, self._postprocess_info(info)

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass
        yield
        pass

    def get_episode_info(self):
        completed_tasks = (
            self.completed_tasks if len(self.completed_tasks) > 0 else [None]
        )
        info = dict(
            solved_subtask=completed_tasks, tasks_to_complete=self.tasks_to_complete
        )
        info.update(self.solved_subtasks)
        return info


class GymWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        from_pixels=False,
        from_state=True,
        height=100,
        width=100,
        camera_id=None,
        channels_first=True,
        frame_skip=1,
        return_state=False,
    ):
        super().__init__(env)
        self._from_pixels = from_pixels
        self._from_state = from_state
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first
        self._frame_skip = frame_skip
        self._return_state = return_state
        if hasattr(self.env, "spec") and self.env.spec is not None:
            if "max_episode_steps" in self.env.spec.kwargs:
                max_episode_steps = self.env.spec.kwargs["max_episode_steps"]
            elif self.env.spec.max_episode_steps:
                max_episode_steps = self.env.spec.max_episode_steps
            else:
                max_episode_steps = self.env.max_episode_steps

        else:
            max_episode_steps = self.env.max_episode_steps
        self.max_episode_steps = max_episode_steps // frame_skip

        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self.observation_space = env.observation_space

        if from_pixels and from_state:
            self.observation_space = gym.spaces.Dict(
                {"image": self.observation_space, "state": env.observation_space}
            )

    def reset(self):
        ob = self.env.reset()

        if self._return_state:
            return self._get_obs(ob, reset=True), ob

        return self._get_obs(ob, reset=True)

    def step(self, ac):
        reward = 0
        for _ in range(self._frame_skip):
            ob, _reward, done, info = self.env.step(ac)
            reward += _reward
            if done:
                break
        if self._return_state:
            return (self._get_obs(ob), ob), reward, done, info

        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob, reset=False):
        state = ob
        if self._from_pixels:
            ob = self.render(
                mode="rgb_array",
                # height=self._height,
                # width=self._width,
                # camera_id=self._camera_id,
            )
            # if reset:
            #     ob = self.render(
            #         mode="rgb_array",
            #         height=self._height,
            #         width=self._width,
            #         camera_id=self._camera_id,
            #     )
            # resize
            ob = cv2.resize(ob, (self._width, self._height))
            if self._channels_first:
                ob = ob.transpose(2, 0, 1).copy()
        else:
            return state

        if self._from_pixels and self._from_state:
            return {"image": ob, "state": state}
        return ob


class DictWrapper(gym.Wrapper):
    def __init__(self, env, return_state=False):
        super().__init__(env)

        self._return_state = return_state

        self._is_ob_dict = isinstance(env.observation_space, gym.spaces.Dict)
        if not self._is_ob_dict:
            self.key = "image" if len(env.observation_space.shape) == 3 else "ob"
            self.observation_space = gym.spaces.Dict({self.key: env.observation_space})
        else:
            self.observation_space = env.observation_space

        self._is_ac_dict = isinstance(env.action_space, gym.spaces.Dict)
        self.action_space = env.action_space

    def reset(self):
        ob = self.env.reset()
        return self._get_obs(ob)

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob):
        if not self._is_ob_dict:
            if self._return_state:
                ob = {self.key: ob[0], "state": ob[1]}
            else:
                ob = {self.key: ob}
        return ob


def stacked_space(space, k):
    if isinstance(space, gym.spaces.Box):
        space_stack = gym.spaces.Box(
            low=np.concatenate([space.low] * k, axis=-1),
            high=np.concatenate([space.high] * k, axis=-1),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        space_stack = gym.spaces.Discrete(space.n * k)
    return space_stack


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=3, return_state=False):
        super().__init__(env)

        # Both observation and action spaces must be gym.spaces.Dict.
        assert isinstance(env.observation_space, gym.spaces.Dict), env.observation_space
        assert isinstance(env.action_space, gym.spaces.Dict), env.action_space

        self._frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)
        self._return_state = return_state
        self._state = None

        ob_space = []
        for k, space in env.observation_space.spaces.items():
            space_stack = stacked_space(space, frame_stack)
            ob_space.append((k, space_stack))
        self.observation_space = gym.spaces.Dict(ob_space)

    def reset(self):
        ob = self.env.reset()
        if self._return_state:
            self._state = ob.pop("state", None)
        for _ in range(self._frame_stack):
            self._frames.append(ob)
        return self._get_obs()

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        if self._return_state:
            self._state = ob.pop("state", None)
        self._frames.append(ob)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        frames = list(self._frames)
        obs = []
        for k in self.env.observation_space.spaces.keys():
            obs.append((k, np.concatenate([f[k] for f in frames], axis=-1)))
        if self._return_state:
            obs.append(("state", self._state))

        return dict(obs)


def zero_value(space, dtype=np.float64):
    if isinstance(space, gym.spaces.Dict):
        return {k: zero_value(space, dtype) for k, space in space.spaces.items()}

    elif isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape).astype(dtype)
    elif isinstance(space, gym.spaces.Discrete):
        return np.zeros(1).astype(dtype)


def get_non_absorbing_state(ob):
    ob = ob.copy()
    ob["absorbing_state"] = np.array([0])
    return ob


def get_absorbing_state(space):
    ob = zero_value(space)
    ob["absorbing_state"] = np.array([1])
    return ob


class AbsorbingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        ob_space = gym.spaces.Dict(spaces=dict(env.observation_space.spaces))
        ob_space.spaces["absorbing_state"] = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.uint8
        )
        self.observation_space = ob_space

    def reset(self):
        ob = self.env.reset()
        return self._get_obs(ob)

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob):
        return get_non_absorbing_state(ob)

    def get_absorbing_state(self):
        return get_absorbing_state(self.observation_space)


def wrap_env(env, cfg):
    env = DictWrapper(env, return_state=False)  # TODO: Do we need this?

    if cfg.pixel_ob and cfg.frame_stack > 1:
        env = FrameStackWrapper(
            env,
            frame_stack=3,
            return_state=cfg.pixel_ob and cfg.state_ob,
        )
    if cfg.absorbing_state:
        env = AbsorbingWrapper(env)

    return env


class GCCalvinWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        goal_image_size: int,
        fixed_goal_image: Optional[
            np.ndarray
        ] = None,  # If None, create image from state
        goal_state: np.ndarray = np.array([0.25, 0.15, 0, 0.088, 1, 1]),
        fixed_reset_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__(env)
        self.env = env
        image_observations: bool = "image" in env.observation_space.spaces
        self.image_observations = image_observations
        self.current_goal = None
        if fixed_goal_image is not None:
            assert len(fixed_goal_image.shape) == 3  # (H, W, C)
            self.fixed_goal_image = fixed_goal_image
        else:
            self.fixed_goal_image = None
        goal_image_size = goal_image_size if image_observations else 200
        self.goal_image_size = goal_image_size
        goal_image_shape = (goal_image_size, goal_image_size, 3)
        self.goal_state = goal_state
        self.fixed_reset_state = fixed_reset_state
        goal_space = (
            gym.spaces.Box(
                low=0,
                high=255,
                shape=goal_image_shape,
                dtype=np.uint8,
            )
            if image_observations
            else env.observation_space["ob"]
        )
        env.observation_space.spaces["image_goal"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=goal_image_shape,
            dtype=np.uint8,
        )
        if not image_observations:
            # Add image observations for video recording
            env.observation_space.spaces["image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=goal_image_shape,
                dtype=np.uint8,
            )
            self.goal_image = None
            self.goal_image_size = 200
        self.observation_space = gym.spaces.Dict(
            dict(
                env.observation_space.spaces,
                goal=goal_space,
                proprio=gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.env.get_obs().shape,
                    dtype=np.float32,
                ),
            )
        )

    def reset(self, **kwargs):
        if self.fixed_reset_state is not None:
            obs = self.env.reset_to_state(*self.fixed_reset_state)
            if self.image_observations:
                obs = {
                    "image": self.render(mode="rgb_array"),
                }
        else:
            obs = self.env.reset(**kwargs)
        if self.image_observations:
            if self.fixed_goal_image is not None:
                goal_image = self.fixed_goal_image
            else:
                goal_image = self.create_goal_image_from_state(
                    scene_obs=self.goal_state
                )

            self.current_goal = goal_image
            obs["image_goal"] = goal_image
        else:
            self.current_goal = obs["ob"].copy()
            self.current_goal[15:21] = self.goal_state
            obs["image"] = self.render(mode="rgb_array")  # For video recording
            self.goal_image = self.create_goal_image_from_state(
                scene_obs=self.goal_state
            )
            obs["image_goal"] = self.goal_image
        obs.update({"goal": self.current_goal, "proprio": self.env.get_obs()})
        return obs

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        if not self.image_observations:
            obs["image"] = self.render(mode="rgb_array")  # For video recording
            obs["image_goal"] = self.goal_image
        else:
            obs["image_goal"] = self.current_goal
        obs.update({"goal": self.current_goal, "proprio": self.env.get_obs()})
        return obs, reward, done, info

    def create_goal_image_from_state(self, scene_obs: np.ndarray):
        current_state = self.env.get_state_obs()

        goal_scene_obs = current_state["scene_obs"].copy()
        goal_scene_obs[:6] = scene_obs

        self.env.reset_to_state(
            robot_obs=current_state["robot_obs"], scene_obs=goal_scene_obs
        )
        env_goal_image = self.env.render(mode="rgb_array")
        env_goal_image = cv2.resize(
            env_goal_image, (self.goal_image_size, self.goal_image_size)
        )
        # Reset the environment to the original state
        self.env.reset_to_state(**current_state)
        return env_goal_image


class AddProprioWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        obs_space = env.observation_space
        proprio_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.env.get_obs().shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            dict(obs_space.spaces, proprio=proprio_space)
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs["proprio"] = self.env.get_obs()
        return obs

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        obs["proprio"] = self.env.get_obs()
        return obs, reward, done, info
