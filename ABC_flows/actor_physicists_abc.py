"""
Implementation of actor pysicists for abc environment. Actions are restricted to just phi values (1d action).
"""


import numpy as np
import torch
import torch.optim as optim
from nn_architectures import PolicyNet


# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            print(f"observed_separation {np.linalg.norm(state_list[-1])}")
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


if __name__=="__main__":
    import matplotlib.pyplot as plt
    #import csv
    from abc_env import ABC_env

    # PARAMS
    A = 1.
    B = 0.7
    C = 0.43
    NU = 0.99
    kappa = 0.001
    BETA = 0.1
    D = 0.1#0.08
    dims = 3
    num_ep = 1000
    t_end = 10.0
#    baseline_phi = 0.4
    
    
    for i in range(11,12):
        baseline_phi = i/10
        env = ABC_env(A,B,C,0.2*np.pi,BETA,kappa,D,NU)
        env.limit = t_end

        stats_actor_loss, rewards_per_episode = [], []
        verbose_episodes = 50
        episode_step_count = int(env.limit/env.deltaT)
        agent = RL_phi_agent(dims)



        # run episodes
        for ep in range(num_ep):
            # history data structs within a episode
            rewards = []
            state = env.getState()
            state_np_arr, next_state_arr = np.array([]).reshape(0, dims), np.array([]).reshape(0, dims)
            # will contain the predicted values according to the environment baseline
            state_val, next_state_val = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)
            action_list, reward_list = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)
            # done_list = [] #not sure if needed
            verbose = ((ep + 1) % verbose_episodes) == 0

            while not env.isOver():
                action = agent.sample_action(state)
                curr_val = env.eval_step(baseline_phi)

                #DEBUGGING
                #action = 0.6

                reward = env.step(action)
                new_val = env.eval_step(baseline_phi)
                
                #print(reward+new_val - curr_val)
                
                
                rewards.append(reward)
                next_state = env.getState()
                state_np_arr = np.vstack((state_np_arr,state))
                action_list = np.vstack((action_list,action))
                reward_list = np.vstack((reward_list,reward))
                next_state_arr = np.vstack((next_state_arr,next_state))
                state_val = np.vstack((state_val,curr_val))
                next_state_val = np.vstack((next_state_val,new_val))
                #done_list.append(1. - env.isOver())


                actor_loss = agent.train(state_np_arr, action_list, reward_list, next_state_arr,state_val, next_state_val, verbose)
                # state_list, actions_list, reward_list, next_state_list, baseline_state, baseline_next_state, verbose=False
                stats_actor_loss.append(actor_loss)
                state = next_state

            rewards_per_episode.append(sum(rewards)/len(rewards))
            print(f"episode {ep} termination distance {env.dist()}")
            env.reset()

        window_size = 50
        avg_policy_loss = []

        # Calculate sliding window averages
        for i in range(len(stats_actor_loss) - window_size + 1):
            window = stats_actor_loss[i : i + window_size]
            window_avg = np.mean(window)
            avg_policy_loss.append(window_avg)


        # post processing (save model, make figs, etc.)



        plt.plot(rewards_per_episode)
        plt.title("Rewards vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig('actor_physics_reward_per_episode.png')

        plt.clf()
        plt.plot(avg_policy_loss)
        plt.savefig('actor_physics_loss.png')
        plt.clf()

        agent.save_policy(f"saved_models/abc_env_{baseline_phi}_{env.limit}_D={D}_eps={num_ep}.pt")

    
    
