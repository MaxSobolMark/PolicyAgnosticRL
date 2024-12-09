import ml_collections

from configs.base_config import (
    BASE_AUTO_REGRESSIVE_TRANSFORMER_CONFIG,
    BASE_DDPM_CONFIG,
    BASE_DIFFUSION_Q_LEARNING_CONFIG,
    BASE_GAUSSIAN_CALQL_CONFIG,
    BASE_PARL_CALQL_CONFIG,
    BASE_PARL_IQL_CONFIG,
    BASE_TRANSFORMER_PARL_CONFIG,
)


def get_config(config_string):
    possible_structures = {
        # Used to pre-train a Diffusion Policy base policy.
        "ddpm": ml_collections.ConfigDict(BASE_DDPM_CONFIG),
        "parl_calql": ml_collections.ConfigDict(BASE_PARL_CALQL_CONFIG),
        "calql": ml_collections.ConfigDict(BASE_GAUSSIAN_CALQL_CONFIG),
        "dql": ml_collections.ConfigDict(BASE_DIFFUSION_Q_LEARNING_CONFIG),
        "parl_iql": ml_collections.ConfigDict(BASE_PARL_IQL_CONFIG),
        "auto_regressive_transformer": ml_collections.ConfigDict(
            BASE_AUTO_REGRESSIVE_TRANSFORMER_CONFIG
        ),
        "transformer_parl": ml_collections.ConfigDict(BASE_TRANSFORMER_PARL_CONFIG),
    }

    return possible_structures[config_string]
