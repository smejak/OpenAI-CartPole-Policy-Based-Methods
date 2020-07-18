import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super().__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.n
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

        self.device = torch.device('cpu')
        
    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

    def act(self, state):
        state = state.unsqueeze(0)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item()
        
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.act(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

    def learn(self, n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
        """PyTorch implementation of the cross-entropy method.
        
        Params
        ======
            n_iterations (int): maximum number of training iterations
            max_t (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
            pop_size (int): size of population at each iteration
            elite_frac (float): percentage of top performers to use in update
            sigma (float): standard deviation of additive noise
        """
        n_elite=int(pop_size*elite_frac) # number of elite policies from the population

        scores_deque = deque(maxlen=100) # list of the past 100 scores
        scores = [] # list of all the scores
        best_weight = sigma*np.random.randn(self.get_weights_dim()) # initialize the first best weight randomly

        for i_iteration in range(1, n_iterations+1): # loop over all the training iterations
            weights_pop = [best_weight + (sigma*np.random.randn(self.get_weights_dim())) for i in range(pop_size)] # population of the weights/policies
            rewards = np.array([self.evaluate(weights, gamma, max_t) for weights in weights_pop]) # rewards from the policies resulting from all individual weights

            # get the best policies
            ##
            elite_idxs = rewards.argsort()[-n_elite:] 
            elite_weights = [weights_pop[i] for i in elite_idxs]
            ##

            best_weight = np.array(elite_weights).mean(axis=0) # take the average of the best weights

            reward = self.evaluate(best_weight, gamma=1.0) # evaluate this new policy
            scores_deque.append(reward) # append the reward
            scores.append(reward) # also append the reward
            
            torch.save(self.state_dict(), './saved_models/checkpoint_cem.pth') # save the agent
            
            if i_iteration % print_every == 0: # print every 100 steps
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

            if np.mean(scores_deque)>=195.0: # print if environment is solved
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
                break
        return scores