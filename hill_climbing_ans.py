import gym
import numpy as np
from collections import deque

class Agent():
    '''
    Agent following a hill-climbing with adaptive noise scaling method
    '''
    def __init__(self, state_size=4, action_size=2):
        '''
        Initialize random weights
        '''
        self.weights = 1e-4*np.random.rand(state_size, action_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.weights)
        return np.exp(x)/sum(np.exp(x))
    
    def act(self, state):
        '''
        Choose an action based on the best policy
        '''
        probs = self.forward(state)
        #action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)              # option 2: deterministic policy
        return action

    def learn(self, env, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
        """Implementation of hill climbing with adaptive noise scaling.
            
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
            noise_scale (float): standard deviation of additive noise
        """
        scores_deque = deque(maxlen=100)
        scores = []
        best_R = -np.Inf
        best_w = self.weights
        for i_episode in range(1, n_episodes+1):
            rewards = []
            state = env.reset()
            for _ in range(max_t):
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break 
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])

            if R >= best_R: # found better weights
                best_R = R
                best_w = self.weights
                noise_scale = max(1e-3, noise_scale / 2)
                self.weights += noise_scale * np.random.rand(*self.weights.shape) 
            else: # did not find better weights
                noise_scale = min(2, noise_scale * 2)
                self.weights = best_w + noise_scale * np.random.rand(*self.weights.shape)

            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                self.weights = best_w
                break
            
        return scores