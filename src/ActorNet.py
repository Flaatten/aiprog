import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from constant.ANETOptimizer import ANETOptimizer
from constant.ActivationFunction import ActivationFunction
from constant.Path import Path


class ActorNet(nn.Module):

    def __init__(self, input_dim, hidden_layers, output_dim, learning_rate, optimizer=ANETOptimizer.SGD, hidden_nodes_activation_function=ActivationFunction.TANH):
        super(ActorNet, self).__init__()
        self.lr = learning_rate
        self.hidden_nodes_activation_function = hidden_nodes_activation_function
        self.layers = self._init_layers(input_dim, hidden_layers, output_dim)
        self.optimizer = self._init_optimizer(optimizer)

    def forward(self, observation):
        x = observation

        # pass through layer, except for the output layer
        for i in range(len(self.layers) - 1):
            if self.hidden_nodes_activation_function == ActivationFunction.LINEAR:
                x = self.layers[i](x)  # TODO CHECK IF THIS IS PROPER WAY TO DO LINEAR
            elif self.hidden_nodes_activation_function == ActivationFunction.RELU:
                x = F.relu(self.layers[i](x))
            elif self.hidden_nodes_activation_function == ActivationFunction.SIGMOID:
                x = F.sigmoid(self.layers[i](x))
            elif self.hidden_nodes_activation_function == ActivationFunction.TANH:
                x = F.tanh(self.layers[i](x))
            else:
                raise ValueError("Invalid activation function provided: " + self.hidden_nodes_activation_function)

        # x is not activated, because it's the last layer. Activate later
        x = self.layers[len(self.layers) - 1](x)
        return x

    def _init_layers(self, input_dim, hidden_layers, output_dim):
        layers = nn.ModuleList()

        layers.append(nn.Linear(input_dim, hidden_layers[0]))  # Input layer

        for i in range(len(hidden_layers) - 1):  # Hidden layers
            layer = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            layers.append(layer)

        layers.append(nn.Linear(hidden_layers[-1], output_dim))  # Output

        return layers

    def _init_optimizer(self, optimizer):
        if optimizer == ANETOptimizer.ADAGRAD:
            return optim.Adagrad(self.parameters(), lr=self.lr)
        elif optimizer == ANETOptimizer.ADAM:
            return optim.Adam(self.parameters(), lr=self.lr)
        elif optimizer == ANETOptimizer.RMSPROP:
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif optimizer == ANETOptimizer.SGD:
            return optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer provided: " + optimizer)

    def train_using_buffer_subset(self, subset):
        self.optimizer.zero_grad()  # zero the gradient buffers
        self.train()

        loss_fn = nn.MSELoss(reduction="mean")
        predicted_y = self(subset.x_train_tensor)

        loss = loss_fn(predicted_y, subset.y_train_tensor)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_params(self, n_iterations):
        torch.save(self.state_dict(), Path.NEURAL_NET + str(n_iterations) + ".model")