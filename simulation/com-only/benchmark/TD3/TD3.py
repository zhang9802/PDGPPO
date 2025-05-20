
import argparse
from collections import namedtuple
from itertools import count
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
import copy

directory = './exp/data.npy'


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, args, device, max_action):
        super(Actor, self).__init__()
        self.device = device


        # hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, args.hidden_dim)
        self.l2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(args.hidden_dim, action_dim)

        # layernorm
        self.ln1 = nn.LayerNorm(state_dim)


        self.max_action = max_action

    def forward(self, state):
        state = self.ln1(state)

        a = F.relu(self.l1(state.float()))

        # Apply batch normalization to the each hidden layer's input
        # a = self.ln2(a)
        a = F.relu(self.l2(a))

        # a = self.ln3(a)

        # a = F.relu(self.l3(a))

        # a = self.ln4(a)
        a = torch.tanh(self.l4(a))


        return self.max_action * a


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		hidden_dim = 512
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		# self.l3 = nn.Linear(hidden_dim, hidden_dim)
		self.l4 = nn.Linear(hidden_dim, 1)
		self.ln1 = nn.LayerNorm(state_dim)
		# self.ln2 = nn.LayerNorm(3)
		# self.ln3 = nn.LayerNorm(128)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, hidden_dim)
		# self.l7 = nn.Linear(hidden_dim, hidden_dim)
		self.l8 = nn.Linear(hidden_dim, 1)



	def forward(self, state, action):
        # state = state.to(torch.float)
        # action = action.to(torch.float)
		state = self.ln1(state)
		sa = torch.cat([state, action], 1)
		# sa = self.ln1(sa)

		q1 = F.relu(self.l1(sa))
		# q1 = self.ln1(q1)

		q1 = F.relu(self.l2(q1))
		# q1 = self.ln2(q1)
		# q1 = F.relu(self.l3(q1))
		# q1 = self.ln3(q1)
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		# q2 = self.ln4(q2)
		q2 = F.relu(self.l6(q2))
		# q2 = self.ln5(q2)
		# q2 = F.relu(self.l7(q2))
		# q2 = self.ln6(q2)
		q2 = self.l8(q2)
		return q1, q2


	def Q1(self, state, action):
		state = self.ln1(state)
		sa = torch.cat([state, action], 1)
		# sa = self.ln1(sa)

		q1 = F.relu(self.l1(sa))
		# q1 = self.ln1(q1)

		q1 = F.relu(self.l2(q1))
		# q1 = self.ln2(q1)
		# q1 = F.relu(self.l3(q1))
		# q1 = self.ln3(q1)
		q1 = self.l4(q1)
		return q1


class TD3():
    def __init__(self, state_dim, action_dim, args, device, max_action=1.0,policy_freq=6, policy_noise=0.1, noise_clip=0.2):

        self.actor = Actor(state_dim, action_dim, args, device, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr, weight_decay=args.decay)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=10*args.lr, weight_decay=args.decay)


        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        # self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        # self.memory = Replay_buffer(args.capacity)
        # self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.discount = args.gamma
        self.tau = args.tau
        self.device = device
        self.total_it = 0
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip



    def select_action(self, state):
        # self.actor.eval()  # model.eval( ) ：不启用 BatchNormalization 和 Dropout
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
        return self.actor.forward(state).cpu().data.numpy().flatten()

    def update_parameters(self, replay_buffer, batch_size=16):
        # self.actor.train()  # model.train( ) ：启用 BatchNormalization 和 Dropout
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # # Select next action according to target policy:
        # noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
        # noise = noise.clamp(-args.noise_clip, args.noise_clip)
        # next_action = (self.actor_target(next_state) + noise)
        # next_action = next_action.clamp(-self.max_action, self.max_action)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)


            # next_action = self.actor_target(next_state)
            # division_term = self.compute_power(next_action.detach())
            # next_action = next_action / division_term
            # next_action = self.actor_target(next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state.to(torch.float), next_action.to(torch.float))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state.to(torch.float), action.to(torch.float))

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates:
        if self.total_it % self.policy_freq == 0:
            # self.actor.train() # model.train( ) ：启用 BatchNormalization 和 Dropout
            # Compute actor loss:
            actor_loss = - self.critic.Q1(state.to(torch.float), self.actor(state).to(torch.float)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

