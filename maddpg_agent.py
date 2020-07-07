# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import DDPG
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'



class MADDPG(object):
    '''The main class that defines and trains all the agents'''
    def __init__(self, state_size, action_size, num_agents, config):
        BUFFER_SIZE = config['BUFFER_SIZE']
        BATCH_SIZE = config['BATCH_SIZE']
        REPLAY_START_EPISODE = config['REPLAY_START_EPISODE']
        RANDOM_SEED = config['RANDOM_SEED']
        GAMMA = config['GAMMA']
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.whole_action_dim = self.action_size*self.num_agents
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED) # Replay memory
        self.maddpg_agents = [DDPG(state_size, action_size, num_agents, config) for _ in range(self.num_agents)] #create agents
        self.replay_start_episode = REPLAY_START_EPISODE
        
    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def step(self, i_episode, states, actions, rewards, next_states, dones):
        #for stepping maddpg
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # index 0 is for agent 0 and index 1 is for agent 1
        all_states = np.reshape(states, newshape=(-1))
        all_next_states = np.reshape(next_states, newshape=(-1))
        
        # Save experience / reward
        self.memory.add(all_states, states, actions, rewards, all_next_states, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and i_episode > self.replay_start_episode:
            for _ in range(NUM_LEARN_PER_STEP): #learn multiple times at every step
                for agent_id in range(self.num_agents):
                    samples = self.memory.sample()
                    
                    self.learn(samples, agent_id, GAMMA)
                self.soft_update_all()

    def soft_update_all(self):
        #soft update all the agents            
        for agent in self.maddpg_agents:
            agent.soft_update_all()
    
    def learn(self, samples, agent_id, gamma):
        #for learning MADDPG
        all_states, states, actions, rewards, all_next_states, next_states, dones = samples
        
        # each actor predicts their own next action based on their next state
        actor_all_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)
        for _id, agent in enumerate(self.maddpg_agents):
            agent_next_state = next_states[:,_id,:]
            actor_all_next_actions[:,_id,:] = agent.actor_target.forward(agent_next_state)
        actor_all_next_actions = actor_all_next_actions.view(-1, self.whole_action_dim)
        
        agent = self.maddpg_agents[agent_id]
        agent_state = states[:,agent_id,:]
        actor_all_actions = actions.clone() #create a deep copy
        actor_all_actions[:,agent_id,:] = agent.actor_local.forward(agent_state)
        actor_all_actions = actor_all_actions.view(-1, self.whole_action_dim)
                
        all_actions = actions.view(-1,self.whole_action_dim)
        
        agent_rewards = rewards[:,agent_id].view(-1,1) #gives wrong result without doing this
        agent_dones = dones[:,agent_id].view(-1,1) #gives wrong result without doing this
        experiences = (all_states, actor_all_actions, all_actions, agent_rewards, \
                       agent_dones, all_next_states, actor_all_next_actions)
        agent.learn(experiences, gamma)

            
    def act(self, all_states, i_episode = 0, add_noise=True):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            action = agent.act(np.reshape(all_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def save_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local_' + str(agent_id) + '.pth')

    def load_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            #Since the model is trained on gpu, need to load all gpu tensors to cpu:
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=\
                                     ["all_state", "state", "action", "reward", "all_next_state", "next_state","done"])
        self.seed = random.seed(seed)
    
    def add(self, all_state, state, action, reward, all_next_state, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(all_state, state, action, reward, all_next_state, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        all_states = torch.from_numpy(np.asarray([e.all_state for e in experiences if e is not None])).float().to(device)
        states = torch.from_numpy(np.asarray([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.asarray([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        all_next_states = torch.from_numpy(np.asarray([e.all_next_state for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.asarray([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (all_states, states, actions, rewards, all_next_states, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)