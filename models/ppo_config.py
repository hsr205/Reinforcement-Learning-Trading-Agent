from dataclasses import dataclass


@dataclass
class PPOConfig:
    observation_dimension: int = 28
    action_dimension: int = 7

    hidden_size: int = 256
    learning_rate: float = 3e-4
    max_gradient_norm: float = 1.0

    max_batch_size: int = 1024
    max_batches: int = 50_000
    sub_batch_size: int = 128
    num_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 1e-4

    cost_coefficient: float = 1e-3
    epsilon: float = 1e-12