import torch.nn as nn


# the neural net to estimate the Q-values
class DQN(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(DQN, self).__init__()
        cnn_out_channels = 8

        # the convolutional part of the network
        """self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )"""

        # the convolutional part of the network
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # two fully-connected layers with <num_actions> outputs to estimate Q-values
        self.value_head = nn.Sequential(
            nn.Linear(board_size * board_size * cnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, state):
        # forward pass through the convolutional layers
        x = self.conv(state)
        # flatten the input and pass it through the fully-connected part
        qvals = self.value_head(x.view(x.size(0), -1))
        return qvals
