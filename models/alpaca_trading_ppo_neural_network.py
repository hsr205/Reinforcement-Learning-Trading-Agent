from collections import defaultdict

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from logger.logger import AppLogger
from models.alpaca_trading_environment_ppo import AlpacaTradingEnvironmentPPO
from models.ppo_config import PPOConfig


class AlpacaTradingPPONeuralNetwork:

    def __init__(self, env: AlpacaTradingEnvironmentPPO, config: PPOConfig) -> None:
        self._config: PPOConfig = config
        self._env: AlpacaTradingEnvironmentPPO = env
        self._device: torch.device = self._env.device
        self._logger = AppLogger.get_logger(self.__class__.__name__)

    def build_actor_module(self) -> ProbabilisticActor:
        actor_network: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=self._config.action_dimension, device=self._device, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, device=self._device, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, device=self._device, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, device=self._device, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, device=self._device, out_features=self._config.action_dimension),
            NormalParamExtractor(),
        )

        # NOTE: Both of the following are used in a Gaussian distribution
        # loc - average (μ)
        # scale - standard deviation (σ)
        actor_backbone: TensorDictModule = TensorDictModule(
            module=actor_network,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        actor_module: ProbabilisticActor = ProbabilisticActor(
            module=actor_backbone,
            spec=self._env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )

        return actor_module

    def build_critic_module(self) -> ValueOperator:
        critic_neural_network: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=self._config.action_dimension, device=self._device, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, device=self._device, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, device=self._device, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, device=self._device, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, device=self._device, out_features=self._config.action_dimension),
        )

        critic_module: ValueOperator = ValueOperator(
            module=critic_neural_network,
            in_keys=["observation"],
        )

        return critic_module

    def train_model(self) -> None:

        actor_module: ProbabilisticActor = self.build_actor_module()
        critic_module: ValueOperator = self.build_critic_module()

        collector: Collector = Collector(
            create_env_fn=self._env,
            policy=actor_module,
            frames_per_batch=self._config.max_batch_size,
            total_frames=self._config.max_batches,
            split_trajs=False,
            device=self._device,
        )

        replay_buffer: ReplayBuffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self._config.max_batch_size),
            sampler=SamplerWithoutReplacement(),
        )

        advantage_module: GAE = GAE(
            gamma=self._config.gamma,
            lmbda=self._config.gae_lambda,
            value_network=critic_module,
            average_gae=True,
            device=self._device,
        )

        loss_module: ClipPPOLoss = ClipPPOLoss(
            actor_network=actor_module,
            critic_network=critic_module,
            clip_epsilon=self._config.epsilon,
            entropy_bonus=True,
            entropy_coef=self._config.entropy_coefficient,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        optimizer = torch.optim.Adam(loss_module.parameters(), lr=self._config.learning_rate)

        logs: dict[str, list[float]] = defaultdict(list)
        progress_bar = tqdm(total=self._config.max_batches)

        for batch_index, tensordict_data in enumerate(collector):
            for _ in range(self._config.num_epochs):
                advantage_module(tensordict_data)

                rollout_view = tensordict_data.reshape(-1)
                replay_buffer.empty()
                replay_buffer.extend(rollout_view.cpu())

                num_mini_batches = self._config.max_batch_size // self._config.sub_batch_size

                for _ in range(num_mini_batches):
                    subdata = replay_buffer.sample(self._config.sub_batch_size)
                    subdata = subdata.to(self._device)

                    loss_values = loss_module(subdata)
                    total_loss = (
                            loss_values["loss_objective"]
                            + loss_values["loss_critic"]
                            + loss_values["loss_entropy"]
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), self._config.max_gradient_norm)
                    optimizer.step()

            mean_reward = float(tensordict_data["next", "reward"].mean().item())
            logs["reward"].append(mean_reward)

            progress_bar.update(tensordict_data.numel())
            progress_bar.set_description(f"batch={batch_index} reward={mean_reward:.6f}")

        self._logger.info("Training finished.")
        self._logger.info(f"Collected {len(logs['reward'])} reward entries.")
