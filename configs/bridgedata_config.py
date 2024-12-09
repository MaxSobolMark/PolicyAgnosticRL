import ml_collections


def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "bridge_v1/berkeley/toykitchen2/?*/train/out.tfrecord",
                        "target_processed/?*/?*.tfrecord",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": None,  # ACTION_PROPRIO_METADATA
            }
        ),
        "pot_to_sink_6dof_10_trajs": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "processed_target_demos_100/pot_to_sink_6dof/train_10_trajs/?*.tfrecord",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": {
                    # These action parameters are used for normalization, so std 0.1 means multiply
                    # actions by 10. We do this because actions from data collection are small, and
                    # this simple scheme seems to work well without needing to calculate dataset
                    # stats.
                    "action": {
                        "mean": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "std": [
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                        ],
                    },
                    # We didn't use proprio for this project.
                    "proprio": {
                        "mean": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "std": [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    },
                },
            }
        ),
        "cup_to_rack_6dof_3distractors_10_trajs": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "processed_target_demos_100/cup_to_rack_6dof_3distractors/train_10/?*.tfrecord",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": {
                    "action": {
                        "mean": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "std": [
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                        ],
                    },
                    "proprio": {
                        "mean": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "std": [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                    },
                },
            }
        ),
    }
    return possible_structures[config_string]
