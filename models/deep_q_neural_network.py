import math
import random
from itertools import count
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import SmoothL1Loss

from logger.logger import AppLogger
from models.replay_memory import ReplayMemory
from utils.constants import Constants

env = gym.make("CartPole-v1")

class DQN(nn.Module):

    def __init__(self, num_observations: int, num_actions: int = 3) -> None:
        super(DQN, self).__init__()
        self._batch_size: int = 64
        self._gamma: float = 0.99
        self._learning_rate = 10e-4
        self._epsilon_end_value: float = 0.01
        self._epsilon_decay_value: int = 2_500
        self._epsilon_start_value: float = 0.9
        self._num_observations = num_observations
        self._update_rate_target_network: float = 0.005
        self._num_actions: int = len(Constants.ACTIONS_LIST)
        self._replay_memory: ReplayMemory = ReplayMemory(10_000)
        self.nn_layer_1 = nn.Linear(in_features=self._num_observations, out_features=64)
        self.nn_layer_2 = nn.Linear(in_features=64, out_features=32)
        self.nn_layer_3 = nn.Linear(in_features=32, out_features=self._num_actions)
        self._device: torch.device = self.get_model_device()
        self._optimizer = optim.AdamW(self.get_policy_network().parameters(), lr=self._learning_rate, amsgrad=True)
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    def forward(self, x):
        layer_1_output = F.relu(self.nn_layer_1(x))
        layer_2_output = F.relu(self.nn_layer_2(layer_1_output))
        return self.nn_layer_3(layer_2_output)

    def get_policy_network(self):
        policy_network: DQN = DQN(self._num_observations, self._num_actions).to(self._device)
        return policy_network

    def get_target_network(self):
        target_network: DQN = DQN(self._num_observations, self._num_actions).to(self._device)
        return target_network

    def get_target_network_state_dict(self):
        policy_network: DQN = self.get_policy_network()
        target_network: DQN = self.get_target_network()

        state_dict = target_network.load_state_dict(policy_network.state_dict())

        return state_dict

    def select_action(self, state):
        steps_num: int = 0
        sample = random.random()
        eps_threshold = self._epsilon_end_value + (self._epsilon_start_value - self._epsilon_end_value) * \
                        math.exp(-1. * steps_num / self._epsilon_decay_value)
        steps_num += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.get_policy_network()(state).max(1).indices.view(1, 1)
        else:
            # TODO: Need to resolve env
            return torch.tensor([[env.action_space.sample()]], device=self._device, dtype=torch.long)

    def optimize_model(self):
        if len(self._replay_memory) < self._batch_size:
            return

        target_network: DQN = self.get_target_network()

        transitions = self._replay_memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self._replay_memory.transition_tuple(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask_tensor: Tensor = torch.tensor(tuple(map(lambda s: s is not None,
                                                               batch.next_state)), device=self._device,
                                                     dtype=torch.bool)
        non_final_next_states_tensor: Tensor = torch.cat([s for s in batch.next_state
                                                          if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.get_policy_network()(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states_tensor are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_tensor: Tensor = torch.zeros(self._batch_size, device=self._device)
        with torch.no_grad():
            next_state_values_tensor[non_final_mask_tensor] = target_network(non_final_next_states_tensor).max(1).values
        # Compute the expected Q values
        expected_state_action_values_tensor: Tensor = (next_state_values_tensor * self._gamma) + reward_batch

        # Compute Huber loss
        criterion: SmoothL1Loss = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values_tensor.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.get_policy_network().parameters(), 100)
        self._optimizer.step()

    def train_agent(self) -> None:

        episode_durations: list[int] = []
        policy_network: DQN = self.get_policy_network()
        target_network: DQN = self.get_target_network()

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            for t in count():
                action = self.select_action(state=state)
                observation, reward_tensor, terminated, truncated, _ = env.step(action.item())
                reward_tensor: Tensor = torch.tensor([reward_tensor], device=self._device)
                is_episode_complete: bool = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self._device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward_tensor)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_network.state_dict()
                policy_net_state_dict = policy_network.state_dict()
                for state in policy_net_state_dict:
                    target_net_state_dict[state] = policy_net_state_dict[state] * self._update_rate_target_network + \
                                                   target_net_state_dict[state] * (
                                                           1 - self._update_rate_target_network)
                target_network.load_state_dict(target_net_state_dict)

                if is_episode_complete:
                    episode_durations.append(t + 1)
                    break

    def get_model_device(self):
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        return device
