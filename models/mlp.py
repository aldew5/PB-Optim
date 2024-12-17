import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 in_features: int = 784, 
                 out_features: int = 1, 
                 hidden_features: int = 300, 
                 init_mean: float = 0, 
                 init_std: float = 4e-2):
        """Initialize a multi-layer perceptron (MLP) model.

        Args:
            in_features (int): Input feature size. Defaults to 784.
            out_features (int): Output feature size. Defaults to 1.
            hidden_features (int): Hidden layer size. Defaults to 300.
            init_mean (float): Weight prior mean. Defaults to 0.
            init_std (float): Weight prior std. Defaults to 4e-2.
        """

        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Define hidden layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

        # Initialize hidden weights from univariate N(init_mean, init_std)
        nn.init.normal_(self.fc1.weight, init_mean, init_std)
        nn.init.normal_(self.fc2.weight, init_mean, init_std)
        nn.init.normal_(self.fc3.weight, init_mean, init_std)

        # Clamp weights to be within 2 standard deviations from the mean
        self.fc1.weight.data = torch.clamp(self.fc1.weight.data, init_mean - 2 * init_std, init_mean + 2 * init_std)
        self.fc2.weight.data = torch.clamp(self.fc2.weight.data, init_mean - 2 * init_std, init_mean + 2 * init_std)
        self.fc3.weight.data = torch.clamp(self.fc3.weight.data, init_mean - 2 * init_std, init_mean + 2 * init_std)

        # Initialize biases
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return (x,)