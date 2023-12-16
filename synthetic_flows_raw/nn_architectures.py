import torch
import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self,input_size,output_size,num_layers, hidden_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x=self.input_layer(x)
        x=self.relu(x)
        for layer in self.hidden_layers:
            x=layer(x)
            x=self.relu(x)
        return self.output_layer(x)

class PolicyNet(nn.Module):
    def __init__(self,input_size,output_size,num_layers,hidden_size, action_scale=1.):
        super().__init__()
        self.input_layer = nn.Linear(input_size,hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.bounding_layer = nn.Tanh()
        self.relu = nn.ReLU()
        self.action_scale = action_scale
        
    def forward(self,x):
        x=self.input_layer(x)
        x=self.relu(x)
        for layer in self.hidden_layers:
            x=layer(x)
            x=self.relu(x)
        #return self.output_layer(x)
        return self.bounding_layer(self.output_layer(x) / self.action_scale) * self.action_scale