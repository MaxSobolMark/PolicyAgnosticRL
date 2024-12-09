import os
from copy import deepcopy

import ml_collections
import tensorflow as tf
from jaxrl_m.agents.continuous.cql import (
    get_default_config as get_continuous_cql_config,
)

from configs.base_config import BASE_PARL_CALQL_CONFIG

DATA_DIR_PREFIX = os.environ.get("DATA_DIR_PREFIX", "./data/")
SAVE_DIR_PREFIX = os.environ.get("SAVE_DIR_PREFIX", "./")


real_robot_params = dict(
    # IP address and port of the robot
    ip="127.0.0.1",
    port=5556,
    # Robot ID
    robot_id=0,
    # General config parameters
    sticky_gripper_num_steps=1,
    env_params=dict(
        camera_topics=[
            dict(
                name="/blue/image_raw",
                flip=False,
            )
        ],
        override_workspace_boundaries=[
            [
                0.25,
                -0.15,
                0.16,
                -0.57,
                0.0,
            ],
            [
                0.33,
                0.11,
                0.29,
                0.57,
                0.0,
            ],
        ],
        move_duration=0.2,
        catch_environment_except=True,
        action_mode="3trans3rot",
        fix_zangle=0.1,
        action_clipping="xyzrot",
    ),
    video_save_path="video_logs",
    shoulder_camera_image_size=256,  # size of image returned by shoulder cam
    initial_eep=[
        0.3,
        0.0,
        0.15,
        0,
        0,
        0,
        1,
    ],
    rollout_timesteps=200,
    image_size=100,
    state_dim=517,
)


def get_config(config_string):
    diffusion_parl_calql_config = deepcopy(BASE_PARL_CALQL_CONFIG)
    diffusion_parl_calql_config["image_observations"] = True
    diffusion_parl_calql_config["encoder"] = "resnetv1-18-bridge"
    diffusion_parl_calql_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    diffusion_parl_calql_config["agent_kwargs"]["cql_alpha"] = 0.01
    diffusion_parl_calql_config["agent_kwargs"]["distributional_critic"] = False
    diffusion_parl_calql_config["agent_kwargs"][
        "critic_network_type"
    ] = "layer_input_mlp"
    diffusion_parl_calql_config["agent_kwargs"]["critic_kwargs"][
        "network_separate_action_input"
    ] = True
    diffusion_parl_calql_config["agent_kwargs"]["critic_network_kwargs"][
        "hidden_dims"
    ] = (
        512,
        512,
        512,
    )
    diffusion_parl_calql_config["agent_kwargs"]["cql_n_actions"] = 4
    diffusion_parl_calql_config["agent_kwargs"]["drq_padding"] = 4
    diffusion_parl_calql_config["base_policy_kwargs"]["image_observations"] = True
    diffusion_parl_calql_config["base_policy_kwargs"]["drq_padding"] = 4
    diffusion_parl_calql_config["dataset_kwargs"] = dict(
        load_dataset_rewards=True,
        goal_relabeling_strategy="robofume_original_rewards",
        goal_relabeling_kwargs=dict(discount=0.99),
        normalization_type="normal",
        dataset_contains_commanded_goals=False,
        augment=False,
        shuffle_buffer_size=25000,
        augment_next_obs_goal_differently=False,
        dof=6,
        relabel_actions=False,
    )
    diffusion_parl_calql_config["general_params"] = real_robot_params

    ddpm_config = deepcopy(BASE_DDPM_CONFIG)
    ddpm_config["image_observations"] = True
    ddpm_config["encoder"] = "resnetv1-18-bridge"
    ddpm_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    ddpm_config["dataset_kwargs"] = dict(
        load_dataset_rewards=True,
        goal_relabeling_strategy="robofume_original_rewards",
        goal_relabeling_kwargs=dict(discount=0.99),
        normalization_type="normal",
        dataset_contains_commanded_goals=False,
        augment=False,
        shuffle_buffer_size=25000,
        augment_next_obs_goal_differently=False,
        dof=6,
        relabel_actions=False,
    )
    ddpm_config["general_params"] = real_robot_params
    ddpm_config["agent_kwargs"]["image_observations"] = True
    ddpm_config["agent_kwargs"]["drq_padding"] = 4

    openvla_parl_calql_config = deepcopy(diffusion_parl_calql_config)
    openvla_parl_calql_config["base_policy_kwargs"] = dict(
        batch_size=16,
        instruction="",
    )
    openvla_parl_calql_config["general_params"]["image_size"] = 224
    # OpenVLA experiments used no demos, only warmups
    openvla_parl_calql_config["mixing_ratio"] = 0.0
    # Can't fit more actions than this on a single h100
    openvla_parl_calql_config["parl_config"]["num_base_policy_actions"] = 8
    openvla_parl_calql_config["parl_config"]["num_actions_to_keep"] = 4

    possible_structures = {
        "diffusion_cql": ml_collections.ConfigDict(diffusion_parl_calql_config),
        "ddpm": ml_collections.ConfigDict(ddpm_config),
        "openvla": ml_collections.ConfigDict(openvla_parl_calql_config),
    }

    return possible_structures[config_string]
