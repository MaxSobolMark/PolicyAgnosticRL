"""
Contains goal relabeling and reward logic written in TensorFlow.

Each relabeling function takes a trajectory with keys "observations",
"next_observations", and "terminals". It returns a new trajectory with the added
keys "goals", "rewards", and "masks". Keep in mind that "observations" and
"next_observations" may themselves be dictionaries, and "goals" must match their
structure.

"masks" determines when the next Q-value is masked out. Typically this is NOT(terminals).
Note that terminal may be changed when doing goal relabeling.
"""

import tensorflow as tf


def uniform(traj, *, reached_proportion, discount, label_mc_returns=True):
    """
    Relabels with a true uniform distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - tf.range(traj_len)

    # add the mc returns
    if label_mc_returns:
        traj["mc_returns"] = (
            -1
            * (tf.math.pow(discount, tf.cast(traj["goal_dists"], tf.float32)) - 1)
            / (discount - 1)
        )

    return traj


def last_state_upweighted(traj, *, reached_proportion, discount, label_mc_returns=True):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range [i + 1, i +
    traj_len). It then gets clipped to be less than traj_len. Therefore, the
    first transition (i = 0) gets a goal sampled uniformly from the future, but
    for i > 0 the last state gets more and more upweighted.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition
    offsets = tf.random.uniform(
        [traj_len],
        minval=1,
        maxval=traj_len,
        dtype=tf.int32,
    )

    # select random transitions to relabel as goal-reaching
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion
    # last transition is always goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # the goal will come from the current transition if the goal was reached
    offsets = tf.where(goal_reached_mask, 0, offsets)

    # convert from relative to absolute indices
    indices = tf.range(traj_len) + offsets

    # clamp out of bounds indices to the last transition
    indices = tf.minimum(indices, traj_len - 1)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, indices),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = indices - tf.range(traj_len)

    # add the mc returns
    if label_mc_returns:
        traj["mc_returns"] = (
            -1
            * (tf.math.pow(discount, tf.cast(traj["goal_dists"], tf.float32)) - 1)
            / (discount - 1)
        )

    return traj


def geometric(
    traj,
    *,
    reached_proportion,
    commanded_goal_proportion=-1.0,
    discount,
    last_state_proportion=0.0,
):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.

    A proportion of goals specified by last_state_proportion will be set to the last state in the trajectory.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = tf.range(traj_len)
    is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
    d = discount ** tf.cast(arange[None] - arange[:, None], tf.float32)

    probs = is_future_mask * d
    # The indexing changes the shape from [seq_len, 1] to [seq_len]
    goal_idxs = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1, dtype=tf.int32
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # select a random proportion of transitions to relabel with the last state
    last_state_mask = tf.random.uniform([traj_len]) < last_state_proportion

    # set goals to the last state for the selected proportion
    goal_idxs = tf.where(last_state_mask, tf.fill([traj_len], traj_len - 1), goal_idxs)

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # Now we will perform the logic for sampling commanded goals
    if commanded_goal_proportion != -1.0:  # if commanded goals are in the dataset
        commanded_goal_mask = tf.random.uniform([traj_len]) < commanded_goal_proportion
        commanded_goal_mask = tf.logical_and(
            commanded_goal_mask, tf.range(traj_len) != traj_len - 1
        )  # last transition must remain goal reaching
        traj["goals"]["image"] = tf.where(
            commanded_goal_mask[..., None, None, None],
            traj["commanded_goals"],
            traj["goals"]["image"],
        )
        traj["rewards"] = tf.where(commanded_goal_mask, -1, traj["rewards"])

    # add masks
    if commanded_goal_proportion != -1.0:
        traj["masks"] = tf.logical_or(
            tf.logical_not(goal_reached_mask), commanded_goal_mask
        )
    else:
        traj["masks"] = tf.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - tf.range(traj_len)

    # We'll also need to add mc_returns
    # Since we have a prior on the reward values, computing discounted sums is easier
    traj["mc_returns"] = 0 * tf.math.pow(
        discount, tf.cast(traj["goal_dists"], tf.float32)
    ) + -1 * (tf.math.pow(discount, tf.cast(traj["goal_dists"], tf.float32)) - 1) / (
        discount - 1
    )
    if commanded_goal_proportion != -1.0:
        traj["mc_returns"] = tf.where(
            commanded_goal_mask, -50.0, traj["mc_returns"]
        )  # hardcoding, but we don't have access to RL discount factor so..

    return traj


def delta_goals(traj, *, goal_delta):
    """
    Relabels with a uniform distribution over future states in the range [i +
    goal_delta[0], min{traj_len, i + goal_delta[1]}). Truncates trajectories to
    have length traj_len - goal_delta[0]. Not suitable for RL (does not add
    terminals or rewards).
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # add the last observation (which only exists in next_observations) to get
    # all the observations
    all_obs = tf.nest.map_structure(
        lambda obs, next_obs: tf.concat([obs, next_obs[-1:]], axis=0),
        traj["observations"],
        traj["next_observations"],
    )
    all_obs_len = traj_len + 1

    # current obs should only come from [0, traj_len - goal_delta[0])
    curr_idxs = tf.range(traj_len - goal_delta[0])

    # select a random future index for each transition i in the range [i + goal_delta[0], min{all_obs_len, i + goal_delta[1]})
    rand = tf.random.uniform([traj_len - goal_delta[0]])
    low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
    high = tf.cast(tf.minimum(all_obs_len, curr_idxs + goal_delta[1]), tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # very rarely, floating point errors can cause goal_idxs to be out of bounds
    goal_idxs = tf.minimum(goal_idxs, all_obs_len - 1)

    traj_truncated = tf.nest.map_structure(
        lambda x: tf.gather(x, curr_idxs),
        traj,
    )

    # select goals
    traj_truncated["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        all_obs,
    )

    # add distances to goal
    traj_truncated["goal_dists"] = goal_idxs - curr_idxs

    return traj_truncated


def robofume(traj, *, discount, num_positive_frames: int = 3):
    """
    Same as robofume. Labels the last num_positive_frames states as positive rewards.
    """
    traj_len = tf.shape(traj["rewards"])[0]
    traj["rewards"] = tf.concat(
        [
            -tf.ones(traj_len - num_positive_frames, dtype=tf.int32),
            tf.zeros(num_positive_frames, dtype=tf.int32),
        ],
        axis=0,
    )
    traj["masks"] = tf.concat(
        [
            tf.ones(traj_len - num_positive_frames, dtype=tf.bool),
            tf.zeros(num_positive_frames, dtype=tf.bool),
        ],
        axis=0,
    )
    traj["terminals"] = tf.logical_not(traj["masks"])
    # Start the range from tf.shape(traj["terminals"])[0] - num_positive_frames
    # because the last num_positive_frames states are positive rewards
    traj["mc_returns"] = tf.concat(
        [
            -1
            * (
                tf.math.pow(
                    discount,
                    tf.cast(
                        tf.range(
                            start=tf.shape(traj["terminals"])[0] - num_positive_frames,
                            limit=-1,
                            delta=-1,
                        ),
                        tf.float32,
                    ),
                )
                - 1
            )
            / (discount - 1),
            tf.zeros(num_positive_frames - 1, dtype=tf.float32),
        ],
        axis=0,
    )
    return traj


def robofume_original_rewards(traj, *, discount):
    """
    Like robofume but gets num_positive_frames from the original rewards.
    Rewards are 1 for the last num_positive_frames states, 0 otherwise.
    """
    traj_len = tf.shape(traj["rewards"])[0]
    num_positive_frames = tf.reduce_sum(tf.cast(traj["rewards"] > 0, tf.int32))
    traj["rewards"] = tf.concat(
        [
            -tf.ones(traj_len - num_positive_frames, dtype=tf.int32),
            tf.zeros(num_positive_frames, dtype=tf.int32),
        ],
        axis=0,
    )
    traj["masks"] = tf.concat(
        [
            tf.ones(traj_len - num_positive_frames, dtype=tf.bool),
            tf.zeros(num_positive_frames, dtype=tf.bool),
        ],
        axis=0,
    )
    traj["terminals"] = tf.logical_not(traj["masks"])
    # Start the range from tf.shape(traj["terminals"])[0] - num_positive_frames
    # because the last num_positive_frames states are positive rewards
    traj["mc_returns"] = tf.concat(
        [
            -1
            * (
                tf.math.pow(
                    discount,
                    tf.cast(
                        tf.range(
                            start=tf.shape(traj["rewards"])[0] - num_positive_frames,
                            limit=-1,
                            delta=-1,
                        ),
                        tf.float32,
                    ),
                )
                - 1
            )
            / (discount - 1),
            tf.zeros(num_positive_frames - 1, dtype=tf.float32),
        ],
        axis=0,
    )
    return traj


GOAL_RELABELING_FUNCTIONS = {
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
    # "last_state": last_state,
    "geometric": geometric,
    "delta_goals": delta_goals,
    "robofume": robofume,
    "robofume_original_rewards": robofume_original_rewards,
}
