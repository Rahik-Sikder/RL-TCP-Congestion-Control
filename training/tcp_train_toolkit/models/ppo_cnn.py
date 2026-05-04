import torch
import torch.nn as nn


class PpoActorCritic1DCNN(nn.Module):
    def __init__(self, k=10, hidden_dim=64):
        super(PpoActorCritic1DCNN, self).__init__()
        self.k = k

        # 1D CNN Feature Extractor
        # Channels = 3 (RTT, dupACK, timeouts), Sequence_length = k (10)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.flatten = nn.Flatten()

        # 32 output channels * k sequence length = 320
        # +1 for the current cwnd scalar
        # +1 for the current loss rate scalar
        cnn_output_dim = (32 * k) + 1 + 1

        # Shared fully connected layer
        self.fc = nn.Sequential(nn.Linear(cnn_output_dim, hidden_dim), self.relu)

        # Actor: Outputs mean and std dev for continuous action [-1, 1]
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic: Estimates cumulative reward (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state shape: (batch_size, 32)

        # Separate time-series from the scalar cwnd and loss rate
        time_series = state[:, :-2]  # Shape: (batch_size, 30)
        cwnd = state[:, -2:-1]  # Shape: (batch_size, 1)
        loss_rate = state[:, -1:]  # Shape: (batch_size, 1)

        # Reshape time-series into 3 channels of length k
        batch_size = state.shape[0]
        time_series = time_series.view(batch_size, 3, self.k)

        # Apply 1D CNN
        x = self.relu(self.conv1(time_series))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)  # Shape: (batch_size, 320)

        # Concatenate CNN features with the global cwnd and loss rate values
        features = torch.cat((x, cwnd, loss_rate), dim=1)  # Shape: (batch_size, 322)

        # Pass through shared dense layer
        hidden = self.fc(features)

        # Critic Value
        value = self.critic(hidden)

        # Actor Action Distribution
        raw_mean = self.actor_mean(hidden).clamp(-10, 10)
        action_mean = torch.tanh(raw_mean)
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))

        return action_mean, action_std, value
