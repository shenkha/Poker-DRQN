import torch
import torch.nn as nn
import numpy as np

class Estimator(object):
    def __init__(
        self,
        num_actions=2,
        learning_rate=0.001,
        state_shape=None,
        mlp_hidden_layer_sizes=None,
        lstm_hidden_size=100,
        device=None,
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device

        self.lstm_input_size = np.prod(self.state_shape)

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(
            lstm_input_size=self.lstm_input_size,
            lstm_hidden_size=self.lstm_hidden_size,
            mlp_hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            mlp_output_size=self.num_actions,
        )

        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction="mean")

        # set up optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, state):
        with torch.no_grad():
            # state = torch.from_numpy(state).float().to(self.device)
            q_as = self.qnet(state).cpu().numpy()
        return q_as

    def update(self, seqs_of_transitions, target_q_values_per_seq):
        self.optimizer.zero_grad()

        self.qnet.train()

        batch_loss = 0

        for i in range(len(seqs_of_transitions)):
            seq = seqs_of_transitions[i]

            self.qnet.reset_hidden_and_cell()
            # basically a sequence of continuos states
            states = np.array([t[0] for t in seq])

            states = (
                torch.FloatTensor(states)
                .view(-1, 1, self.lstm_input_size)
                .to(self.device)
            )

            actions = torch.LongTensor([t[1] for t in seq]).view(-1, 1).to(self.device)
            target = (
                torch.FloatTensor(target_q_values_per_seq[i]).view(-1).to(self.device)
            )

            # (batch, state_shape) -> (batch, num_actions)
            q_as = self.qnet(states)

            # (batch, num_actions) -> (batch, )

            Q = (
                torch.gather(q_as, dim=-1, index=actions.unsqueeze(-1))
                .squeeze(-1)
                .view(-1)
            )
            batch_loss += self.mse_loss(Q, target)

        # update model
        batch_loss.backward()
        self.optimizer.step()

        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss

    def checkpoint_attributes(self):
        """Return the attributes needed to restore the model from a checkpoint"""
        return {
            "qnet": self.qnet.state_dict(),
            "lstm_hidden_state":self.qnet.lstm_hidden_state,
            "cell_state":self.qnet.cell_state,
            "optimizer": self.optimizer.state_dict(),
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "state_shape": self.state_shape,
            "mlp_hidden_layer_sizes": self.mlp_hidden_layer_sizes,
            "lstm_hidden_size": self.lstm_hidden_size,
            "device": self.device,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Restore the model from a checkpoint"""
        estimator = cls(
            num_actions=checkpoint["num_actions"],
            lstm_hidden_size=checkpoint["lstm_hidden_size"],
            learning_rate=checkpoint["learning_rate"],
            state_shape=checkpoint["state_shape"],
            mlp_hidden_layer_sizes=checkpoint["mlp_hidden_layer_sizes"],
            device=checkpoint["device"],
        )

        estimator.qnet.load_state_dict(checkpoint["qnet"])
        estimator.qnet.load_hidden_and_cell(checkpoint["lstm_hidden_state"], checkpoint["cell_state"])
        estimator.optimizer.load_state_dict(checkpoint["optimizer"])
        return estimator


class EstimatorNetwork(nn.Module):
    def __init__(
        self,
        lstm_input_size,  # number: state size
        lstm_hidden_size,  # number: lstm output size
        mlp_hidden_layer_sizes,  # array
        mlp_output_size,  # number of actions
    ):
        super(EstimatorNetwork, self).__init__()

        self.lstm_input_size = lstm_input_size  # state size
        self.lstm_hidden_size = lstm_hidden_size  # lstm output size
        self.lstm_num_layers = 1

        self.lstm = nn.LSTM(
            self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers
        )

        # init the hidden state and the cell state
        self.reset_hidden_and_cell()

        # mlp input: self.lstm_hidden_size

        self.mlp_input_size = self.lstm_hidden_size

        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes

        self.mlp_output_size = mlp_output_size  # number of actions

        all_layer_sizes = (
            [self.mlp_input_size] + self.mlp_hidden_layer_sizes + [self.mlp_output_size]
        )

        fc = []

        for l in range(len(all_layer_sizes) - 1):
            fc.append(nn.Linear(all_layer_sizes[l], all_layer_sizes[l + 1]))

            if l < len(all_layer_sizes) - 1:
                # if it is not the last layer
                fc.append(nn.Tanh())

        self.mlp = nn.Sequential(*fc)

    def forward(self, input):
        if self.lstm_hidden_state is None and self.cell_state is None:
            x, (self.lstm_hidden_state, self.cell_state) = self.lstm(input)
        else:
            x, (self.lstm_hidden_state, self.cell_state) = self.lstm(
                input, (self.lstm_hidden_state, self.cell_state)
            )
        return self.mlp(x)

    def reset_hidden_and_cell(self):
        self.lstm_hidden_state = None
        self.cell_state = None

    def load_hidden_and_cell(self, hidden, cell):
        if hidden == None:
            self.lstm_hidden_state = None
        else:
            self.lstm_hidden_state = torch.clone(hidden)

        if cell == None:
            self.cell_state = None
        else:
            self.cell_state = torch.clone(cell)
