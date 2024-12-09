import os
from copy import deepcopy

import ml_collections
import tensorflow as tf
from jaxrl_m.agents.continuous.cql import (
    get_default_config as get_continuous_cql_config,
)
from jaxrl_m.agents.continuous.diffusion_q_learning import (
    get_default_config as get_diffusion_q_learning_config,
)

from configs.base_config import (
    BASE_DDPM_CONFIG,
    BASE_DIFFUSION_Q_LEARNING_CONFIG,
    BASE_GAUSSIAN_CALQL_CONFIG,
    BASE_PARL_CALQL_CONFIG,
)

SAVE_DIR_PREFIX = os.environ.get("SAVE_DIR_PREFIX", "./")


def get_config(config_string):
    parl_calql_config = deepcopy(BASE_PARL_CALQL_CONFIG)
    parl_calql_config["save_video"] = True
    parl_calql_config["image_observations"] = True
    parl_calql_config["encoder"] = "resnetv1-18-bridge"
    parl_calql_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    parl_calql_config["calvin_tfrecord_regexp"] = (
        "./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord"
    )
    parl_calql_config["dataset_kwargs"] = dict(
        cache=False,
        tfrecords_include_next_observations=False,
    )
    parl_calql_config["agent_kwargs"]["cql_alpha"] = 0.01
    parl_calql_config["agent_kwargs"]["distributional_critic"] = False
    parl_calql_config["agent_kwargs"]["critic_network_type"] = "layer_input_mlp"
    parl_calql_config["agent_kwargs"]["critic_kwargs"][
        "network_separate_action_input"
    ] = True
    parl_calql_config["agent_kwargs"]["critic_network_kwargs"]["hidden_dims"] = (
        512,
        512,
        512,
    )
    parl_calql_config["agent_kwargs"]["cql_n_actions"] = 4
    parl_calql_config["agent_kwargs"]["drq_padding"] = 4
    parl_calql_config["base_policy_agent_kwargs"]["image_observations"] = True
    parl_calql_config["base_policy_agent_kwargs"]["drq_padding"] = 4
    parl_calql_config["distill_argmax"] = True
    parl_calql_config["image_replay_buffer_kwargs"] = dict()

    ddpm_config = deepcopy(BASE_DDPM_CONFIG)
    ddpm_config["save_video"] = True
    ddpm_config["image_observations"] = True
    ddpm_config["encoder"] = "resnetv1-18-bridge"
    ddpm_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    ddpm_config["calvin_tfrecord_regexp"] = (
        "./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord"
    )
    ddpm_config["dataset_kwargs"] = dict(
        cache=False,
        tfrecords_include_next_observations=False,
    )
    ddpm_config["agent_kwargs"]["image_observations"] = True
    ddpm_config["agent_kwargs"]["drq_padding"] = 4

    gaussian_calql_config = deepcopy(BASE_GAUSSIAN_CALQL_CONFIG)
    gaussian_calql_config["save_video"] = True
    gaussian_calql_config["image_observations"] = True
    gaussian_calql_config["encoder"] = "resnetv1-18-bridge"
    gaussian_calql_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    gaussian_calql_config["calvin_tfrecord_regexp"] = (
        "./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord"
    )
    gaussian_calql_config["dataset_kwargs"] = dict(
        cache=False,
        tfrecords_include_next_observations=False,
    )
    gaussian_calql_config["agent_kwargs"]["distributional_critic"] = False
    gaussian_calql_config["agent_kwargs"]["critic_network_type"] = "layer_input_mlp"
    gaussian_calql_config["agent_kwargs"]["critic_kwargs"][
        "network_separate_action_input"
    ] = True
    gaussian_calql_config["agent_kwargs"]["critic_network_kwargs"]["hidden_dims"] = (
        512,
        512,
        512,
    )
    gaussian_calql_config["agent_kwargs"]["cql_n_actions"] = 4
    gaussian_calql_config["agent_kwargs"]["drq_padding"] = 4

    dql_config = deepcopy(BASE_DIFFUSION_Q_LEARNING_CONFIG)
    dql_config["save_video"] = True
    dql_config["image_observations"] = True
    dql_config["encoder"] = "resnetv1-18-bridge"
    dql_config["encoder_kwargs"] = dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    )
    dql_config["calvin_tfrecord_regexp"] = (
        "./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord"
    )
    dql_config["dataset_kwargs"] = dict(
        cache=False,
        tfrecords_include_next_observations=False,
    )
    dql_config["agent_kwargs"]["distributional_critic"] = False
    dql_config["agent_kwargs"]["critic_network_type"] = "layer_input_mlp"
    dql_config["agent_kwargs"]["critic_kwargs"]["network_separate_action_input"] = True
    dql_config["agent_kwargs"]["critic_network_kwargs"]["hidden_dims"] = (
        512,
        512,
        512,
    )
    dql_config["agent_kwargs"]["drq_padding"] = 4

    possible_structures = {
        "parl_calql": ml_collections.ConfigDict(parl_calql_config),
        "ddpm": ml_collections.ConfigDict(ddpm_config),
        "gaussian_calql": ml_collections.ConfigDict(gaussian_calql_config),
        "dql": ml_collections.ConfigDict(dql_config),
    }

    return possible_structures[config_string]
