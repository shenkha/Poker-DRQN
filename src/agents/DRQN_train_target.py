import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import datetime
from .models.DRQN_model import Estimator

class DRQN_target_Agent(object):
    def __init__(
        self,
        num_actions=2,
        state_shape=None,
        mlp_layers=None,
        lstm_hidden_size=100,
        device=None,
    ) -> None:
        self.use_raw = False
        self.num_actions = num_actions
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers

        self.state_shape = state_shape

        self.lstm_input_size = state_shape[0]
        print(self.lstm_input_size)

        # Torch device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        self.q_net = Estimator(
            num_actions=self.num_actions,
            lstm_hidden_size=self.lstm_hidden_size,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device,
        )


    
    def step(self, state):
        q_values = self.predict(state)

        return np.argmax(q_values)

    def eval_step(self, state):
        """Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        """
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info["values"] = {
            state["raw_legal_actions"][i]: float(
                q_values[list(state["legal_actions"].keys())[i]]
            )
            for i in range(len(state["legal_actions"]))
        }

        return best_action, info

    def predict(self, state):
        legal_actions = list(state["legal_actions"].keys())

        state = state["obs"]
        state = (
            torch.tensor(state, dtype=torch.float32)
            .view(-1, self.lstm_input_size)
            .to(self.device)
        )
        q_values = self.q_net.predict_nograd(state)[0]

        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)

        for a in legal_actions:
            masked_q_values[a] = q_values[a]

        return masked_q_values

    def set_device(self, device):
        self.device = device
        self.q_net.device = device

    def reset_hidden_and_cell(self):
        self.q_net.qnet.reset_hidden_and_cell()
