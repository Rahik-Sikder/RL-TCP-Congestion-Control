import torch
import torch.nn as nn


class DdpgActorCritic1DCNN(nn.Module):
    def __init__(self, k=10, hidden_dim=64):
        super(DdpgActorCritic1DCNN, self).__init__()
        self.k = k

        # Actor network
        self.actor_conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.actor_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.actor_relu = nn.ReLU()
        self.actor_flatten = nn.Flatten()
        actor_input_dim = (32 * k) + 1 + 1
        self.actor_fc = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic network
        self.critic_conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.critic_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.critic_relu = nn.ReLU()
        self.critic_flatten = nn.Flatten()
        critic_input_dim = (32 * k) + 1 + 1 + 1  # CNN features + cwnd + action + loss rate
        self.critic_fc = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _extract_time_series_and_cwnd(self, state):
        time_series = state[:, :-2]
        cwnd = state[:, -2:-1]
        loss_rate = state[:, -1:]
        batch_size = state.shape[0]
        time_series = time_series.view(batch_size, 3, self.k)
        return time_series, cwnd, loss_rate

    def actor(self, state):
        time_series, cwnd, loss_rate = self._extract_time_series_and_cwnd(state)
        x = self.actor_relu(self.actor_conv1(time_series))
        x = self.actor_relu(self.actor_conv2(x))
        x = self.actor_flatten(x)
        features = torch.cat((x, cwnd, loss_rate), dim=1)
        hidden = self.actor_fc(features)
        action = torch.tanh(self.actor_head(hidden))
        return action

    def critic(self, state, action):
        time_series, cwnd, loss_rate = self._extract_time_series_and_cwnd(state)
        x = self.critic_relu(self.critic_conv1(time_series))
        x = self.critic_relu(self.critic_conv2(x))
        x = self.critic_flatten(x)
        features = torch.cat((x, cwnd, action, loss_rate), dim=1)
        hidden = self.critic_fc(features)
        q_value = self.critic_head(hidden)
        return q_value

    def forward(self, state):
        # Forward returns deterministic actor action for inference/export.
        return self.actor(state)

    def actor_parameters(self):
        return list(self.actor_conv1.parameters()) + list(self.actor_conv2.parameters()) + \
            list(self.actor_fc.parameters()) + list(self.actor_head.parameters())

    def critic_parameters(self):
        return list(self.critic_conv1.parameters()) + list(self.critic_conv2.parameters()) + \
            list(self.critic_fc.parameters()) + list(self.critic_head.parameters())
