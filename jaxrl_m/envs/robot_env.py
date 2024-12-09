import time
from typing import Any, Callable, Dict, Optional

import gym
import numpy as np
import torch
from jaxrl_m.envs.robot_manual_reward_functions import CVDiscriminator
from PIL import Image


def get_observation(widowx_client, config):
    while True:
        obs = widowx_client.get_observation()
        if obs is None:
            print("WARNING: failed to get robot observation, retrying...")
        else:
            break
    obs["image"] = (
        obs["image"]
        .reshape(
            3,
            config["general_params"]["shoulder_camera_image_size"],
            config["general_params"]["shoulder_camera_image_size"],
        )
        .transpose(1, 2, 0)
        * 255
    ).astype(np.uint8)
    return obs


def get_reward_function(
    task_name: str, task_encoding: np.ndarray, dof: int
) -> Callable[[Dict[str, np.ndarray], np.ndarray], float]:
    task_info = {
        "task_names": [task_name],
        "task_ids": [task_encoding],
        "goal_frames": None,
    }
    cv_discriminator = CVDiscriminator(task_info)

    def reward_function(state: Dict[str, np.ndarray], action: np.ndarray) -> float:
        image = state["image"]
        if image.shape != (100, 100, 3):
            image = np.array(Image.fromarray(state["image"]).resize((100, 100))).astype(
                np.uint8
            )
        assert state["proprio"].shape == (512 + dof + 1,)
        gripper_state = state["proprio"][dof]
        return cv_discriminator(
            images=torch.tensor(image[None].transpose(0, 3, 1, 2)),
            states=torch.tensor(
                np.concatenate(
                    [gripper_state[None], state["proprio"][dof + 1 :]], axis=0
                )
            )[None],
        )[0].item()

    return reward_function


def preprocess_observation(
    obs: Dict[str, Any],
    env_params,
    language_conditioning_encoding,
    image_size: int = 100,
) -> Dict[str, Any]:
    obs["image"] = np.array(
        Image.fromarray(obs["image"]).resize(
            (
                image_size,
                image_size,
            )
        )
    ).astype(np.uint8)
    assert obs["state"].shape == (7,)
    action_mode = env_params["action_mode"]
    if action_mode == "3trans1rot":
        obs["state"] = np.concatenate([obs["state"][:3], obs["state"][5:]])
    elif action_mode == "3trans":
        obs["state"] = np.concatenate([obs["state"][:3], obs["state"][6:]])
    obs["state"] = np.concatenate([obs["state"], language_conditioning_encoding])
    obs = {
        "image": obs["image"],
        "proprio": obs["state"],
    }
    return obs


def get_robot_action_space(
    normalization_action_mean: np.ndarray,
    normalization_action_std: np.ndarray,
    dof: int,
) -> gym.spaces.Box:
    original_action_space_low = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25])[
        :dof
    ]
    original_action_space_high = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25])[:dof]
    return gym.spaces.Box(
        np.concatenate(
            [
                (original_action_space_low - normalization_action_mean[:dof])
                / normalization_action_std[:dof],
                [0],
            ]
        ),
        np.concatenate(
            [
                (original_action_space_high - normalization_action_mean[:dof])
                / normalization_action_std[:dof],
                [1],
            ]
        ),
    )


class RobotEnv(gym.Env):
    def __init__(
        self,
        env_params,
        ip,
        port,
        train_config,
        reward_function,
        initial_eep,
        language_conditioning_encoding,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        image_size: int = 100,
        termination_reward_function_name: Optional[str] = None,
    ):
        from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

        self.env_params = WidowXConfigs.DefaultEnvParams.copy()
        self.env_params.update(env_params)
        self.widowx_client = WidowXClient(
            host=ip,
            port=port,
        )
        self.train_config = train_config
        self.reward_function = reward_function
        self.image_size = image_size

        if env_params["action_mode"] == "3trans1rot":
            self.dof = 4
        elif env_params["action_mode"] == "3trans":
            self.dof = 3
        elif env_params["action_mode"] == "3trans3rot":
            self.dof = 6
        else:
            raise ValueError("Invalid action mode")

        if termination_reward_function_name is not None:
            task_encoding = language_conditioning_encoding
            self.termination_reward_function = get_reward_function(
                task_name=termination_reward_function_name,
                task_encoding=task_encoding,
                dof=self.dof,
            )
        else:
            self.termination_reward_function = None

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=np.zeros((image_size, image_size, 3)),
                    high=255 * np.ones((image_size, image_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=float("-inf") * np.ones(512 + self.dof + 1),
                    high=float("inf") * np.ones(512 + self.dof + 1),
                    dtype=np.float32,
                ),
            }
        )

        self.last_action_time = None
        self.initial_eep = initial_eep
        self.language_conditioning_encoding = language_conditioning_encoding
        action_mean = action_mean[:6]
        action_std = action_std[:6]
        if self.dof == 4:
            action_mean = np.concatenate([action_mean[:3], [action_mean[-2]]])
            action_std = np.concatenate([action_std[:3], [action_std[-2]]])
        elif self.dof == 3:
            action_mean = action_mean[:3]
            action_std = action_std[:3]
        self.action_mean = action_mean
        self.action_std = action_std
        self.num_steps = 0

        self.action_space = get_robot_action_space(
            normalization_action_mean=self.action_mean,
            normalization_action_std=self.action_std,
            dof=self.dof,
        )

        self.widowx_client.init(self.env_params)

    def move_to_start_state(self, eep_pos):
        successful = False
        while not successful:
            try:
                # Get XYZ position from user.
                x_val, y_val, z_val = eep_pos
                # x_val = input(f"Enter x value of gripper starting position (leave empty for default == {init_x}): ")
                # if x_val == "":
                #    x_val = init_x
                # y_val = input(f"Enter y value of gripper starting position (leave empty for default == {init_y}): ")
                # if y_val == "":
                #    y_val = init_y
                # z_val = input(f"Enter z value of gripper starting position (leave empty for default == {init_z}): ")
                # if z_val == "":
                #    z_val = init_z

                # Fix initial orientation and add user's commanded XYZ into start transform.
                # Initial orientation: gripper points ~15 degrees away from the standard orientation (quat=[0, 0, 0, 1]).
                transform = np.array(
                    [
                        [0.267, 0.000, 0.963, float(x_val)],
                        [0.000, 1.000, 0.000, float(y_val)],
                        [-0.963, 0.000, 0.267, float(z_val)],
                        [0.00, 0.00, 0.00, 1.00],
                    ]
                )
                # IMPORTANT: It is very important to move to reset position with blocking==True.
                #            Otherwise, the controller's `_reset_previous_qpos()` call will be called immediately after
                #            the move command is given -- and before the move is complete -- and the initial state will
                #            be totally incorrect.
                self.widowx_client.move(transform, duration=0.8, blocking=True)
                successful = True
            except Exception as e:
                print(e)

    def reset(self):

        # To make sure human hand doesnt cause distribution shift
        input("Press enter to reset environment...")

        print("Resetting environment...")

        self.widowx_client.reset()
        print(f"Moving to position {self.initial_eep[:3]}")
        self.move_to_start_state(self.initial_eep[:3])

        obs = get_observation(self.widowx_client, self.train_config)
        obs = preprocess_observation(
            obs,
            self.env_params,
            self.language_conditioning_encoding,
            image_size=self.image_size,
        )

        self.num_steps = 0

        return obs, {}

    def step(self, action):
        if self.action_mean is not None:
            action = action.copy()
            action = np.clip(action, self.action_space.low, self.action_space.high)
            action[: self.dof] = action[: self.dof] * self.action_std + self.action_mean
        else:
            assert False

        if self.last_action_time is not None:
            action_per_second = 1 / (time.time() - self.last_action_time)
            # time.sleep(max(0, 0.17 - (time.time() - self.last_action_time)))
        else:
            action_per_second = None
        self.widowx_client.step_action(action, blocking=False)
        self.last_action_time = time.time()
        obs = get_observation(self.widowx_client, self.train_config)
        print(f"get_observation time: {time.time() - self.last_action_time}")
        done = obs["env_done"]
        obs = preprocess_observation(
            obs,
            self.env_params,
            self.language_conditioning_encoding,
            image_size=self.image_size,
        )
        if self.termination_reward_function is not None:
            done = done or bool(self.termination_reward_function(obs, action))

        reward = self.reward_function(obs, action)

        self.num_steps += 1
        print(f"Step {self.num_steps}, reward: {reward}, aps: {action_per_second}")

        return obs, reward, done, {}
