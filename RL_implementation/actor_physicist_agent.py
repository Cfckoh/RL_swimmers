import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#_______________________ Policy NN Architecture _______________________

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
        return self.bounding_layer(self.output_layer(x) / self.action_scale) * self.action_scale
    



#_______________________ Actor Physicists Agent _______________________

class RL_phi_agent:
    def __init__(self, state_size, output_size=2, hidden_size=32, num_layers=3, actor_lr=0.00001, discount=.99, action_scale=5.):
        self.actor_net = PolicyNet(input_size=state_size, output_size=output_size, num_layers=num_layers, hidden_size=hidden_size, action_scale=action_scale)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.discount = discount

    def sample_action(self, state):
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
            output_params = self.actor_net(input_state)
            phi = torch.normal(mean=output_params[0], std = torch.exp(output_params[1]))
            phi = phi.detach().cpu().numpy()

        
        return phi
    

    def sample_deterministic_action(self,state):
        input_state = torch.FloatTensor(state).to(device)
        output_params = self.actor_net(input_state)
        phi = output_params[0].detach().cpu().numpy()
        return phi

    def train(self, state_list, actions_list, reward_list, next_state_list, baseline_state, baseline_next_state, verbose=False):
        """possibly needed params: 
            - time remiaing (add to state?)
            - done_list?

        """
        #create tensors
        state_t = torch.FloatTensor(state_list).to(device)
        #next_state_t = torch.FloatTensor(next_state_list).to(device)
        actions_t = torch.FloatTensor(actions_list).to(device)
        reward_t = torch.FloatTensor(reward_list).to(device)
        baseline_t = torch.FloatTensor(baseline_state).to(device)
        next_baseline_t = torch.FloatTensor(baseline_next_state).to(device)


        # NOTE may need to discount the reward list (I don't think you do actually)
        approx_value = baseline_t
        sampled_q_value = reward_t + next_baseline_t
        advantage_t = sampled_q_value - approx_value
        
        # DEBUG
        #advantage_t = reward_t
        
        outputs = self.actor_net(state_t)
        # testing phi is actually sampled
        distributions = torch.distributions.Normal(outputs[:,0], torch.exp(outputs[:,1]))
        log_probs = distributions.log_prob(actions_t)

        actor_loss = torch.mean(-log_probs * advantage_t)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 


        if verbose:
            print(f"observed_separation {torch.linalg.norm(state_t[-1])}")
            print(f"actual phi {actions_t[-1]}")
            print(f"mean: {outputs[:,0][-1]}")
            #print(f"log prob: {log_probs[-1]}")
            print(f"advantage: {advantage_t[-1]}")
            print(f"stds: {torch.exp(outputs[:,1][-1])}")
            #print(f"state approx value: {approx_value[-1]}")
            #print(f"next state approx value: {bootstrap_value[-1]}")
            print()


        return actor_loss.detach().cpu().numpy()

    def save_policy(self, file_name):
        """
        Save policy for evaluation cannot be trained further when saving this way.
        """
        torch.save(self.actor_net.state_dict(), file_name)


    def load_policy(self, file_name):
        """
        Load policy for evaluation cannot be trained further
        """
        self.actor_net.load_state_dict(torch.load(file_name))
        self.actor_net.eval()