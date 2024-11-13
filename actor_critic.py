import torch
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, tau, learning_rate):
        super(ActorNetwork, self).__init__()
        self.tau = tau
        self.learning_rate = learning_rate
        
        # Define the actor network architecture
        self.model = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 600),
            nn.ReLU(),
            nn.Linear(600, action_size),
            nn.Tanh()  # Output should be in the range [-1, 1]
        )
        
        # Target network
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 600),
            nn.ReLU(),
            nn.Linear(600, action_size),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, state):
        return self.model(state)
    
    def train_actor(self, states, action_grads):
        actions = self.forward(states)
        actor_loss = -torch.mean(action_grads * actions)
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
    
    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, tau, learning_rate):
        super(CriticNetwork, self).__init__()
        self.tau = tau
        self.learning_rate = learning_rate
        
        # Define the critic network architecture
        self.model = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 600),
            nn.ReLU()
        )
        
        self.action_layer = nn.Linear(action_size, 600)
        self.output_layer = nn.Linear(600 * 2, 1)  # Change to 1 for Q-value output
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Target network
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 600),
            nn.ReLU()
        )
        self.target_action_layer = nn.Linear(action_size, 600)
        self.target_output_layer = nn.Linear(600 * 2, 1)  # Change to 1 for Q-value output
    
    def forward(self, state, action):
        state_out = self.model(state)
        action_out = self.action_layer(action)
        
        combined = torch.cat((state_out, action_out), dim=1)
        return self.output_layer(combined)
    
    def train_critic(self, states, actions, q_targets):
        q_values = self.forward(states, actions)
        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_action_layer.parameters(), self.action_layer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_output_layer.parameters(), self.output_layer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def target_forward(self, state, action):
        state_out = self.target_model(state)
        action_out = self.target_action_layer(action)
        combined = torch.cat((state_out, action_out), dim=1)
        return self.target_output_layer(combined)
