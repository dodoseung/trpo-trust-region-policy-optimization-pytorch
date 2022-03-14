# Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. PMLR, 2015.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
import gym
from collections import deque

class ReplayBuffer():
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.memory = []
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # Reset the replay memory
    def reset(self):
        self.memory = []
        
class ContinousPolicyNet(nn.Module):
    def __init__(self, state_num, min_action, max_action):
        super(ContinousPolicyNet, self).__init__()
        self.min_action = min_action
        self.max_action = max_action
        
        self.input = nn.Linear(state_num, 32)
        self.mu = nn.Linear(32, 1)
        self.std = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        mu = (self.max_action - self.min_action) * F.sigmoid(self.mu(x)) + self.min_action
        std = (self.max_action - self.min_action) * F.sigmoid(self.std(x)) / 2

        return mu, std

class CriticNet(nn.Module):
    def __init__(self, state_num):
        super(CriticNet, self).__init__()
        self.input = nn.Linear(state_num, 32)
        self.output = nn.Linear(32, 1)
    
    def forward(self, x):      
        x = F.relu(self.input(x))
        value = self.output(x)
        return value

class TRPO():
    def __init__(self, env, gamma=0.99, learning_rate=1e-3, delta=0.05):
        super(TRPO, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_min = float(env.action_space.low[0])
        self.action_max = float(env.action_space.high[0])
     
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy (actor)
        self.actor_net = ContinousPolicyNet(self.state_num, self.action_min, self.action_max).to(self.device)
        
        # Critic
        self.critic_net = CriticNet(self.state_num).to(self.device)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        
        # Rollout
        self.memory = ReplayBuffer()
        
        # Learning setting
        self.gamma = gamma
        
        # Constraint
        self.delta = delta
        
    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu, std = self.actor_net(state)
        action = D.Normal(mu, std).sample()
        action = action.cpu().detach().numpy()
        
        return action[0]
    
    # Hessian-vector product
    def hvp(self, d_kl, v, params, retain_graph):
        return self.flat_grad(d_kl @ v, params, retain_graph)

    # Conjugate gradient to calculate Ax = b
    def conjugate_gradient(self, A, d_kl, params, retain_graph, b, max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone() # b - Ax
        v = r.clone() # r

        for _ in range(max_iterations):
            Av = A(d_kl, v, params, retain_graph)
            alpha = (r @ r) / (v @ Av)
            x_new = x + alpha * v   
            r = r - alpha * Av
            v = r - (r @ Av) / (v @ Av) * v
            x = x_new
            
        return x
    
    # Surrogate objective for maximizing
    def surrogate_objective(self, log_prob_old, log_prob_new, advantages):
        objective = advantages * torch.exp(log_prob_new - log_prob_old)
        return objective.mean()
    
    # KL divergence
    def kl_divergence(self, mu_old, std_old, logstd_old, mu_new, std_new, logstd_new):
        kl = (logstd_old - logstd_new) + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2)) - 0.5
        return kl.sum()
    
    # Flatten a gradient
    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        retain_graph = True if create_graph == True else retain_graph
        grad = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        grad = torch.cat([t.view(-1) for t in grad])
        return grad
    
    # Update a parameter from flattend gradient
    def param_update(self, policy_net, flattened_grad):
        index = 0
        for param in policy_net.parameters():
            param_length = param.numel()
            grad = flattened_grad[index : index+param_length].view(param.shape)
            param.data += grad
            index += param_length
    
    def learn(self):
        # Get memory from rollout
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states[-1]).to(self.device)
        done = dones[-1]
        
        # Critic network
        values = self.critic_net(states)
        next_value = self.critic_net(next_state)
        
        # Calculate target values and advantages
        R = [0] * (actions.size(dim=0) + 1)
        R[-1] = next_value if not done else 0
        for i in reversed(range(len(R)-1)):
            R[i] = rewards[i] + self.gamma * R[i+1]
        R = torch.FloatTensor(R[:-1]).to(self.device).view(-1,1)
        
        # Calculate and normalize advantages to reduce skewness and improve convergence
        advantages = R.detach() - values
        advantages = ((advantages - advantages.mean()) / advantages.std()).view(1, -1) if len(advantages) > 1 else advantages

        # Calculate critic losses and optimize the critic network
        critic_loss = 0.5 * advantages.pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Get pi theta old
        mu_old, std_old = self.actor_net(states)
        dist_old = D.Normal(mu_old, std_old)
        log_probs_old = dist_old.log_prob(actions)
        
        # Compute L and KL
        L_old = self.surrogate_objective(log_probs_old.detach(), log_probs_old, advantages)
        KL_old = self.kl_divergence(mu_old.detach(), std_old.detach(), log_probs_old.detach(), mu_old, std_old, log_probs_old)
        
        # Policy network parameters
        params = list(self.actor_net.parameters())

        # Set the g and kl gradient
        g = self.flat_grad(L_old, params, retain_graph=True)
        d_kl = self.flat_grad(KL_old, params, create_graph=True)
        
        # s ia a search direction and beta is a maximal step length
        s = self.conjugate_gradient(self.hvp, d_kl, params, True, g)
        beta = torch.sqrt(2 * self.delta / (s @ self.hvp(d_kl, s, params, True)))
        max_step = beta * s

        # Line search
        for i in range(10):
            # Set the step size
            step = (0.9 ** i) * max_step
            
            # Apply parameters' update
            self.param_update(self.actor_net, step)

            with torch.no_grad():            
                # Get pi theta new after updating the network
                mu_new, std_new = self.actor_net(states)
                dist_new = D.Normal(mu_new, std_new)
                log_probs_new = dist_new.log_prob(actions)
                
                # Compute L and KL after updating the network
                L_new = self.surrogate_objective(log_probs_old.detach(), log_probs_new, advantages)
                KL_new = self.kl_divergence(mu_old.detach(), std_old.detach(), log_probs_old.detach(), mu_new, std_new, log_probs_new)

            # Calculate the improvement of objective value
            L_improvement = L_new - L_old
            
            # If the improvement of L is positive and the kl value is lower than delta, fix the parameters
            if L_improvement > 0 and KL_new <= self.delta:
                break
            
            # Else, reset the parameters
            self.param_update(self.actor_net, -step)
            
        # Reset the memory
        self.memory.reset()
            
        
def main():
    env = gym.make("Pendulum-v0")
    agent = TRPO(env, gamma=0.99, learning_rate=1e-6, delta=0.01)
    ep_rewards = deque(maxlen=20)
    total_episode = 10000

    for i in range(total_episode):
        state = env.reset()
        rewards = []

        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)

            agent.memory.add(state, action, reward, next_state, done)             
            rewards.append(reward)
            
            if done:
                agent.learn()
                ep_rewards.append(sum(rewards))
                
                if i % 20 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state
    

if __name__ == '__main__':
    main()