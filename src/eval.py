import rlcard
import os.path
import torch
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils.utils import tournament
from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard import models

from agents.DRQN_agent import DRQNAgent

env = rlcard.make("limit-holdem")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)
print("device:", device)


drqn_agent = DRQNAgent.from_checkpoint(
    torch.load("saves/checkpoint_drqn340000.pt"), save_path="saves", save_every=10000
)
drqn_agent.reset_hidden_and_cell()

random_agent = RandomAgent(num_actions=env.num_actions)
pretrained_model = models.load("limit-holdem-rule-v1").agents[0]

env.set_agents([drqn_agent, pretrained_model])
# env.set_agents([drqn_agent, random_agent])

play_num = 1000
win = 0
min_score = 10000
max_score = -10000
sum_re = 0
for i in range(play_num):
    drqn_agent.reset_hidden_and_cell()
    rewards = tournament(env, 1)
    for position, reward in enumerate(rewards):
        if position == 0:
            # print(reward)
            min_score = min(min_score, reward)
            max_score = max(max_score, reward)
            sum_re += reward
            if reward > 0:
                win += 1

print("win rate: ", win, "/", play_num)
print("min score: ", min_score)
print("max score: ", max_score)
print("sum of reward: ", sum_re)
