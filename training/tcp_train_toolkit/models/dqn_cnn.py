import torch
import torch.nn as nn


class DqnQNetwork1DCNN(nn.Module):
    def __init__(self, k=10, hidden_dim=64, num_actions=6):
        super(DqnQNetwork1DCNN, self).__init__()
        self.k = k
        self.num_actions = num_actions

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

        cnn_output_dim = (32 * k) + 1

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            self.relu,
        )
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        time_series = state[:, :-1]
        cwnd = state[:, -1:]

        batch_size = state.shape[0]
        time_series = time_series.view(batch_size, 3, self.k)

        x = self.relu(self.conv1(time_series))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        features = torch.cat((x, cwnd), dim=1)
        hidden = self.fc(features)
        q_values = self.q_head(hidden)
        return q_values
