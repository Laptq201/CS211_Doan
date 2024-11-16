import torch
import torch.nn as nn
import gym
import numpy as np
from models import RLNN
import time

# Define the Actor class without layer normalization


class Actor(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        # Layer normalization is disabled based on saved model
        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.max_action = max_action

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

# Define Args class with layer_norm set to False


class Args:
    layer_norm = False  # Disable layer normalization to match saved model
    actor_lr = 0.001
    tau = 0.005
    discount = 0.99


args = Args()

# Set up environment and actor with correct dimensions
env = gym.make("Ant-v4", render_mode="human")
state_dim = 27  # Updated state dimension to match environment
action_dim = 8  # Action dimension from saved model
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action, args)
actor.load_state_dict(torch.load('/home/lapquang/Downloads/2/actor(1).pkl'))
actor.eval()

# Function to render the agent's actions in the environment


def render_agent(actor, env, max_steps=10000):
    state = env.reset()[0]  # Updated for compatibility
    done = False
    steps = 0
    total_reward = 0

    while not done:  # not done:  # and steps < max_steps:
        env.render()
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action = np.clip(action, -max_action, max_action)
        state, reward, done, *_ = env.step(action)
        total_reward += reward
        steps += 1
        time.sleep(0.05)

    env.close()
    print("Total Reward:", total_reward)


# Render the agent
render_agent(actor, env)
