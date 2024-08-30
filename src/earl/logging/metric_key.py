import enum


class MetricKey(enum.StrEnum):
    """Keys for the metrics that will be returned by GymnaxLoop.run().

    Note that agents can return additional metrics.
    """

    ACTION_COUNTS = enum.auto()
    DURATION_SEC = enum.auto()
    LOSS = enum.auto()
    TOTAL_REWARD = enum.auto()
    TOTAL_DONES = enum.auto()
    """Mean over num_envs."""
    REWARD_MEAN_SMOOTH = enum.auto()
    """Mean over num_envs."""
    COMPLETE_EPISODE_LENGTH_MEAN = enum.auto()
    NUM_ENVS_THAT_DID_NOT_COMPLETE = enum.auto()
    STEP_NUM = enum.auto()
