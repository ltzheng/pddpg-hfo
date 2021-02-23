import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CentralCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, param_dim, hidden_layers=None, action_input_layer=0,
                 init_type="normal", activation="leaky_relu", init_std=0.01, num_agents=1):
        super(CentralCritic, self).__init__()
        self.obs_dim = obs_dim * num_agents
        self.act_dim = act_dim * num_agents
        self.param_dim = param_dim * num_agents
        self.activation = activation

        # self.action_input_layer = action_input_layer

        # initialise layers
        self.layers = nn.ModuleList()
        input_size = self.obs_dim + self.act_dim + self.param_dim
        last_hidden_layer_size = input_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_layer_size = hidden_layers[nh - 1]
        self.output_layer = nn.Linear(last_hidden_layer_size, 1)

        # initialise layers
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight.data, nonlinearity=self.activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError("Unknown init_type " + str(init_type))
            nn.init.zeros_(self.layers[i].bias.data)
        nn.init.normal_(self.output_layer.weight, std=init_std)
        # nn.init.normal_(self.action_output_layer.bias, std=init_std)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, obs, actions, action_parameters):
        x = torch.cat((obs, actions, action_parameters), dim=1)
        negative_slope = 0.01

        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.output_layer(x)

        return Q
