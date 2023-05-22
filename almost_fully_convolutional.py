import torch
from torch import nn
from yaml import load as load, CLoader as CLoader


class FullyConvolutionLogistic(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 125 * output_size)

        self.conv1 = nn.Conv1d(1, 8, 5, 5)
        self.conv2 = nn.Conv1d(8, 16, 5, 5)
        self.conv3 = nn.Conv1d(16, 32, 5, 5)

        self.fc2 = nn.Linear(32 * output_size, output_size)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # BATCH x 12500
        x = x.unsqueeze(1)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.fc2(x)
        return x


class DenseLogistic(nn.Module):
    activation_map = {'RELU': nn.ReLU()}

    def __init__(self, yaml_path):
        """Specify a dense NN model for the logistic coefficient map from a given yaml file"""

        super().__init__()

        with open(yaml_path) as file_handle:
            self.model_spec = load(file_handle, Loader=CLoader)

        for (i, layer) in enumerate(self.model_spec.get('layers', [])):
            setattr(self, f'fc{1 + i}', nn.Linear(layer['input_size'], layer['output_size']))
            activation = layer.get('activation', None)
            if activation is not None:
                setattr(self, f'activation{1 + i}', self.activation_map[activation])

    def forward(self, x):
        """Evaluate the model"""

        for (i, layer) in enumerate(self.model_spec.get('layers', [])):
            x = getattr(self, f'fc{1 + i}')(x)
            activation = layer.get('activation', None)
            if activation is not None:
                x = getattr(self, f'activation{1 + i}')(x)
        return x
