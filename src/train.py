import signal
import rlcard
import os.path
import torch
from rlcard.agents import RandomAgent, DQNAgent
from logger import logger
from rlcard.utils.utils import tournament
from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from agents.DRQN_agent import DRQNAgent
from agents.DRQN_train_target import DRQN_target_Agent


# Make environments to train and evaluate models
env = rlcard.make("limit-holdem")
eval_env = rlcard.make("limit-holdem")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Number of actions:", env.num_actions)
print("Number of players:", env.num_players)
print("Shape of state:", env.state_shape)
print("Shape of action:", env.action_shape)
print("device:", device)


target_update_frequency = 700
max_epsilon = 1
min_epsilon = 0.1
epsilon_decay_steps = 10000
gamma = 0.99  # discount_factor
lr = 0.00005
memory_size = 20000
min_replay_size = 200
batch_size = 32
num_actions = env.num_actions
state_shape = env.state_shape[0]
train_every = 1
mlp_layers = [256, 512]
lstm_hidden_size = 128
save_path = "saves"
save_every = 5000


eval_every = 500
eval_num = 100
episode_num = 300000

update_every = 50

# initialize DRQN target agents
train_target = DRQN_target_Agent(
    num_actions=num_actions,
    state_shape=state_shape,
    mlp_layers=mlp_layers,
    lstm_hidden_size=lstm_hidden_size,
    device=device,
)

if os.path.isfile(save_path + "/last.pt"):
    drqn_agent = DRQNAgent.from_checkpoint(
        torch.load(save_path + "/last.pt"), save_path=save_path, save_every=save_every
    )

    train_target.q_net.qnet.load_state_dict(drqn_agent.q_net.qnet.state_dict())
    drqn_agent.reset_hidden_and_cell()
    train_target.reset_hidden_and_cell()
else:
    drqn_agent = DRQNAgent(
        target_update_frequency=target_update_frequency,
        max_epsilon=max_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_steps=epsilon_decay_steps,
        gamma=gamma,  # discount_factor
        lr=lr,
        memory_size=memory_size,
        min_replay_size=min_replay_size,
        batch_size=batch_size,
        num_actions=num_actions,
        state_shape=state_shape,
        train_every=train_every,
        mlp_layers=mlp_layers,
        lstm_hidden_size=lstm_hidden_size,
        save_path=save_path,
        save_every=save_every,
        device=device,
    )
    train_target.q_net.qnet.load_state_dict(drqn_agent.q_net.qnet.state_dict())
    drqn_agent.reset_hidden_and_cell()
    train_target.reset_hidden_and_cell()


# pretrained_model = models.load("limit-holdem-rule-v1").agents[0]

random_agent = RandomAgent(num_actions=eval_env.num_actions)

eval_env.set_agents([drqn_agent, random_agent])

env.set_agents([drqn_agent, train_target])


def handler(signum, frame):
    print("\n\tSIGINT received\n\tsaving model and shutting down")
    drqn_agent.save_checkpoint(filename="last.pt")
    exit()


signal.signal(signal.SIGINT, handler)


for episode in range(episode_num):
    if episode % update_every == 0 and drqn_agent.memory.ready():
        # print("\rINFO - copy model parameters to target agent")
        train_target.q_net.from_checkpoint(drqn_agent.q_net.checkpoint_attributes())
    # reset hidden state of recurrent agents
    drqn_agent.reset_hidden_and_cell()
    train_target.reset_hidden_and_cell()

    # get transitions by playing an episode in env
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    drqn_agent.feed(trajectories[0])

    if episode % eval_every == 0:
        score = 0
        for i in range(eval_num):
            drqn_agent.reset_hidden_and_cell()

            score += tournament(eval_env, 1)[0]
        logger.add_scalar(
            "reward vs. random agent", score / eval_num, drqn_agent.total_t
        )
