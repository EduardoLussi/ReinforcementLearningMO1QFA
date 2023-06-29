import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.device = "cuda" if T.cuda.is_available() else "cpu"        
        self.to(self.device)

        time_file = time.strftime("%m%d-%H%M") + ("_actor" if n_actions == 2 else "_critic")
        self.checkpoint_dir = f'tmp/actor_critic'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, time_file)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
