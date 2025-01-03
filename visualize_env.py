# import torch
# import torch.nn as nn
# import gym
# import numpy as np
# from models import RLNN
# import time

# # Define the Actor class without layer normalization


# # class Actor(RLNN):
# #     def __init__(self, state_dim, action_dim, max_action, args):
# #         super(Actor, self).__init__(state_dim, action_dim, max_action)
# #         self.l1 = nn.Linear(state_dim, 400)
# #         self.l2 = nn.Linear(400, 300)
# #         self.l3 = nn.Linear(300, action_dim)

# #         # Layer normalization is disabled based on saved model
# #         self.layer_norm = args.layer_norm
# #         self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
# #         self.tau = args.tau
# #         self.discount = args.discount
# #         self.max_action = max_action

# #     def forward(self, x):
# #         x = torch.tanh(self.l1(x))
# #         x = torch.tanh(self.l2(x))
# #         x = self.max_action * torch.tanh(self.l3(x))
# #         return x


# class Actor(RLNN):

#     def __init__(self, state_dim, action_dim, max_action, args):
#         super(Actor, self).__init__(state_dim, action_dim, max_action)

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, action_dim)

#         if args.layer_norm:
#             self.n1 = nn.LayerNorm(400)
#             self.n2 = nn.LayerNorm(300)
#         self.layer_norm = args.layer_norm

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
#         self.tau = args.tau
#         self.discount = args.discount
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_action = max_action

#     def forward(self, x):

#         if not self.layer_norm:
#             x = torch.tanh(self.l1(x))
#             x = torch.tanh(self.l2(x))
#             x = self.max_action * torch.tanh(self.l3(x))

#         else:
#             x = torch.tanh(self.n1(self.l1(x)))
#             x = torch.tanh(self.n2(self.l2(x)))
#             x = self.max_action * torch.tanh(self.l3(x))

#         return x

#     def update(self, memory, batch_size, critic, actor_t):

#         # Sample replay buffer
#         states, _, _, _, _ = memory.sample(batch_size)

#         # Compute actor loss
#         if args.use_td3:
#             actor_loss = -critic(states, self(states))[0].mean()
#         else:
#             actor_loss = -critic(states, self(states)).mean()

#         # Optimize the actor
#         self.optimizer.zero_grad()
#         actor_loss.backward()
#         self.optimizer.step()

#         # Update the frozen target models
#         for param, target_param in zip(self.parameters(), actor_t.parameters()):
#             target_param.data.copy_(
#                 self.tau * param.data + (1 - self.tau) * target_param.data)



# # Define Args class with layer_norm set to False


# class Args:
#     layer_norm = False  # Disable layer normalization to match saved model
#     actor_lr = 0.001
#     tau = 0.005
#     discount = 0.99


# args = Args()

# # Set up environment and actor with correct dimensions
# env = gym.make("Ant-v4", render_mode="human")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])
# # state_dim = 27  # Updated state dimension to match environment
# # action_dim = 8  # Action dimension from saved model
# # max_action = float(env.action_space.high[0])

# actor = Actor(state_dim, action_dim, max_action, args)
# actor.load_state_dict(torch.load('/home/lapquang/Downloads/Ant/CEM_TD3.pkl', weights_only=True)) # Updated path to saved model
# actor.eval()

# # Function to render the agent's actions in the environment


# def render_agent(actor, env, max_steps=300):
#     state = env.reset()[0]  # Updated for compatibility
#     done = False
#     steps = 0
#     total_reward = 0

#     while not done and steps < max_steps:
#         env.render()
#         state_tensor = torch.FloatTensor(state.reshape(1, -1))
#         action = actor(state_tensor).cpu().data.numpy().flatten()
#         action = np.clip(action, -max_action, max_action)
#         state, reward, done, *_ = env.step(action)
#         total_reward += reward
#         steps += 1
#         time.sleep(0.05)

#     env.close()
#     print("Total Reward:", total_reward)


# # Render the agent
# render_agent(actor, env)
import torch
import torch.nn as nn
import gymnasium as gym  # Import gymnasium instead of gym
import numpy as np
from models import RLNN
import time
from gymnasium.wrappers import RecordVideo  # Correct wrapper for video recording

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
env = gym.make("Ant-v4", render_mode='rgb_array')  # Updated render_mode to 'rgb_array'
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Wrap the environment with RecordVideo to save video
env = RecordVideo(env, './video', episode_trigger=lambda episode_id: True)

actor = Actor(state_dim, action_dim, max_action, args)
actor.load_state_dict(torch.load('/home/lapquang/Downloads/actor_0 (1).pkl', weights_only=True))
actor.eval()

# Function to render the agent's actions in the environment
def render_agent(actor, env, max_steps=1000):
    state = env.reset()[0]  # Updated for compatibility
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:  # not done:  # and steps < max_steps:
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action = np.clip(action, -max_action, max_action)
        state, reward, done, _, _ = env.step(action)  # Updated for compatibility
        total_reward += reward
        steps += 1


    env.close()
    print("Total Reward:", total_reward)

# Render the agent and save video
render_agent(actor, env)
