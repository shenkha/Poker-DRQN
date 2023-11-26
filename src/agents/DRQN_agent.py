import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from logger import logger
from .models.DRQN_model import Estimator


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done", "legal_actions"]
)


class Memory:
    def __init__(self, memory_size, min_replay_size, batch_size) -> None:
        self.memory_size = memory_size
        self.min_replay_size = min_replay_size

        self.batch_size = batch_size

        self.memory = []

    def ready(self):
        if len(self.memory) >= self.min_replay_size:
            return True
        return False

    def add(self, experience):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def checkpoint_attributes(self):
        """Returns the attributes that need to be checkpointed"""

        return {
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "memory": self.memory,
            "min_replay_size": self.min_replay_size,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """
        Restores the attributes from the checkpoint

        Args:
            checkpoint (dict): the checkpoint dictionary

        Returns:
            instance (Memory): the restored instance
        """

        instance = cls(
            checkpoint["memory_size"],
            checkpoint["min_replay_size"],
            checkpoint["batch_size"],
        )
        instance.memory = checkpoint["memory"]
        return instance


class DRQNAgent(object):
    def __init__(
        self,
        target_update_frequency=1000,
        max_epsilon=1,
        min_epsilon=0.1,
        epsilon_decay_steps=20000,
        gamma=0.99,  # discount_factor
        lr=0.00005,
        memory_size=10000,
        min_replay_size=100,
        batch_size=32,
        num_actions=2,
        state_shape=None,
        train_every=1,
        mlp_layers=None,
        lstm_hidden_size=100,
        device=None,
        save_path="saves",
        save_every=float("inf"),
    ) -> None:
        self.use_raw = False
        self.memory_size = memory_size
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

        self.min_replay_size = min_replay_size

        self.lr = lr

        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers

        self.state_shape = state_shape

        self.lstm_input_size = state_shape[0]

        # Torch device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(min_epsilon, max_epsilon, epsilon_decay_steps)

        self.q_net = Estimator(
            num_actions=self.num_actions,
            lstm_hidden_size=self.lstm_hidden_size,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device,
        )

        self.target_net = Estimator(
            num_actions=self.num_actions,
            lstm_hidden_size=self.lstm_hidden_size,
            learning_rate=self.lr,
            state_shape=self.state_shape,
            mlp_hidden_layer_sizes=self.mlp_layers,
            device=self.device,
        )

        self.target_net.qnet.load_state_dict(self.q_net.qnet.state_dict())

        self.memory = Memory(
            memory_size=memory_size,
            min_replay_size=min_replay_size,
            batch_size=batch_size,
        )

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def feed(self, seq_of_transition):
        if len(seq_of_transition) == 0:
            return
        seq = []

        for ts in seq_of_transition:
            (state, action, reward, next_state, done) = tuple(ts)

            seq.append(
                Transition(
                    state=state["obs"],
                    action=action,
                    reward=reward,
                    next_state=next_state["obs"],
                    legal_actions=list(next_state["legal_actions"].keys()),
                    done=done,
                )
            )

        self.memory.add(seq)
        self.total_t += 1

        if (
            self.memory.ready()
            and (self.total_t - self.min_replay_size) % self.train_every == 0
        ):
            self.train()

    def step(self, state):
        """Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        """
        q_values = self.predict(state)

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]

        legal_actions = list(state["legal_actions"].keys())

        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)

        best_action_idx = legal_actions.index(np.argmax(q_values))

        probs[best_action_idx] += 1.0 - epsilon
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

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

    def train(self):
        """Train the network

        Returns:
            loss (float): The loss of the current batch.
        """
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()
        sequences = self.memory.sample()

        # calculate the q value of each state in each seq

        target_q_values_per_seq = []

        for seq in sequences:
            self.target_net.qnet.reset_hidden_and_cell()

            next_states = np.array([t[3] for t in seq])
            rewards = np.array([t[2] for t in seq])
            dones = np.array([t[4] for t in seq])

            next_states = (
                torch.FloatTensor(next_states)
                .view(-1, 1, self.lstm_input_size)
                .to(self.device)
            )

            q_values_next_target = self.target_net.predict_nograd(next_states)

            # Compute targets using the formulation sample = r + gamma * max q(s',a')
            max_target_q_values = q_values_next_target.max(axis=-1).reshape((-1))

            q_values_target = rewards + self.gamma * (1 - dones) * max_target_q_values

            target_q_values_per_seq.append(q_values_target)

        loss = self.q_net.update(sequences, target_q_values_per_seq)

        if self.train_t % 50 == 0:
            # save losses to tensorboard
            logger.add_scalar("loss per 50 updates", loss, self.total_t)

        if self.train_t % 20 == 0:
            print("\rINFO - Step {}, rl-loss: {}".format(self.total_t, loss), end="")

        # Update the target estimator
        if self.train_t % self.target_update_frequency == 0:
            # self.target_net = deepcopy(self.q_net)
            self.target_net.qnet.load_state_dict(self.q_net.qnet.state_dict())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately,
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(
                self.save_path,
                filename="checkpoint_drqn" + str(self.train_t) + ".pt",
            )
            print("\nINFO - Saved model checkpoint.")

    def set_device(self, device):
        self.device = device
        self.q_net.device = device
        self.target_net.device = device

    def reset_hidden_and_cell(self):
        self.q_net.qnet.reset_hidden_and_cell()
        self.target_net.qnet.reset_hidden_and_cell()

    def checkpoint_attributes(self):
        """
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        """

        return {
            "agent_type": "DQNAgent",
            "q_net": self.q_net.checkpoint_attributes(),
            "memory": self.memory.checkpoint_attributes(),
            "total_t": self.total_t,
            "train_t": self.train_t,
            "target_update_frequency": self.target_update_frequency,
            "max_epsilon": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "gamma": self.gamma,
            "lr": self.lr,
            "num_actions": self.num_actions,
            "train_every": self.train_every,
            "device": self.device,
            "save_path": self.save_path,
            "save_every": self.save_every,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, save_path=None, save_every=None):
        """
        Restore the model from a checkpoint

        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        """

        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            target_update_frequency=checkpoint["target_update_frequency"],
            max_epsilon=checkpoint["max_epsilon"],
            min_epsilon=checkpoint["min_epsilon"],
            epsilon_decay_steps=checkpoint["epsilon_decay_steps"],
            gamma=checkpoint["gamma"],  # discount_factor
            lr=checkpoint["lr"],
            memory_size=checkpoint["memory"]["memory_size"],
            min_replay_size=checkpoint["memory"]["min_replay_size"],
            batch_size=checkpoint["memory"]["batch_size"],
            num_actions=checkpoint["num_actions"],
            state_shape=checkpoint["q_net"]["state_shape"],
            train_every=checkpoint["train_every"],
            mlp_layers=checkpoint["q_net"]["mlp_hidden_layer_sizes"],
            lstm_hidden_size=checkpoint["q_net"]["lstm_hidden_size"],
            device=checkpoint["device"],
            save_path=save_path if save_path is not None else checkpoint["save_path"],
            save_every=save_every if save_every is not None else checkpoint["save_every"],
        )

        agent_instance.total_t = checkpoint["total_t"]
        agent_instance.train_t = checkpoint["train_t"]

        agent_instance.q_net = Estimator.from_checkpoint(checkpoint["q_net"])
        agent_instance.target_net.qnet.load_state_dict(
            agent_instance.q_net.qnet.state_dict()
        )
        agent_instance.memory = Memory.from_checkpoint(checkpoint["memory"])

        return agent_instance

    def save_checkpoint(self, path="saves", filename="checkpoint_drqn.pt"):
        """Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
        """
        torch.save(self.checkpoint_attributes(), path + "/" + filename)
