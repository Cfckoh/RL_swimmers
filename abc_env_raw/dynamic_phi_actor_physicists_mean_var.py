import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from nn_architectures import PolicyNet
import copy

# hyper that may want to tune later
BASELINE_SCALE = 1.
SEPERATION_DISCOUNT = 1. #0.1#1.5#0.25

class Physics_baseline:
    def __init__(self, prescribed_phi, lyapunov_mu, lyapunov_sigma, sample_time, dimension=3, cost_coef=1, cost_power=2):
        self.prescribed_phi = prescribed_phi
        self.lyapunov_mu = lyapunov_mu
        self.cramer_d2 = 1/(lyapunov_sigma * sample_time)
        self.dimension = dimension
        self.cost_coef = cost_coef
        self.cost_power = cost_power
        
    def evaluate_prescribed(self, seperation):
        """Evaluate the physics baseline given a certain seperation"""
        # neeed to make this a vecotrized operation


        first_term_num = self.prescribed_phi**2*seperation**2
        first_term_denom = self.dimension + 2*(self.prescribed_phi-self.lyapunov_mu)*self.cramer_d2-2
        second_term_num = self.cost_coef*seperation**self.cost_power
        second_term_denom = self.dimension + 2*(self.prescribed_phi-self.lyapunov_mu)*self.cramer_d2-self.cost_power
        return -BASELINE_SCALE*(first_term_num/first_term_denom + second_term_num/second_term_denom)

    def evaluate_do_nothing(self, seperation, time_remaining):
        return -self.cost_coef*(seperation**self.cost_power/(self.lyapunov_mu*self.cost_power))*(np.exp(self.cost_power*self.lyapunov_mu*time_remaining) - 1)
    
    def mixed_evaluation(self, seperation, time_remaining, do_nothing_weight):
        # do_nothing weight should be between 1 and 0 and the weight of the prescribed motion is 1-do_nothing_weight

        return (1-do_nothing_weight)*self.evaluate_prescribed(seperation) + do_nothing_weight*self.evaluate_do_nothing(seperation, time_remaining)



class Baseline_reworked:
    def __init__(self, prescribed_phi,lyapunov_mu, lyapunov_sigma, sample_time, r_d,cost_coef=1, nu = .025):
        self.lyapunov = lyapunov_mu
        self.prescribed_phi = prescribed_phi
        self.cramer_d2 = 1/(lyapunov_sigma**2 * sample_time)
        self.beta = cost_coef
        self.r_d = r_d
        self.nu = nu # ??? this doesn't seem right
        self.a = self.lyapunov + 1/self.cramer_d2
        #self.b =  self.lyapunov + 3/(2*self.cramer_d2)
        #print(self.a)
    
    def evaluate_prescribed(self, seperation):#,steps_remaining):
        "RETURNS THE PRESCRIBED PENALTY SHOULD BE A NEGATIVE"
        a = self.a
        #b = self.b
        first_term = (seperation**2/self.lyapunov)*np.log(seperation/self.r_d)
        second_term = (self.r_d**2*(self.prescribed_phi-self.lyapunov))/(self.nu*(self.prescribed_phi-a)) #* steps_remaining/4
        #print(first_term)
        #print(second_term)
        first_term = torch.clamp(first_term,min=0.0)
        return -(self.prescribed_phi**2 + self.beta) * (first_term * SEPERATION_DISCOUNT + second_term)



# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicPhiAgent:
    def __init__(self, state_size=3, hidden_size=64, num_layers=4, actor_lr=0.00001, discount=1., action_scale=1.):
        self.actor_net = PolicyNet(input_size=state_size, output_size=2, num_layers=num_layers, hidden_size=hidden_size, action_scale=action_scale)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.discount = discount


    def sample_action(self, state):
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
            output_params = self.actor_net(input_state)
            #print(torch.exp(output_params[1])) 
            phi = torch.normal(mean=output_params[0], std = torch.exp(output_params[1]))
            phi = phi.detach().cpu().numpy()

        
        return state*phi

    def train(self, state_list, selected_phi_list, reward_list, next_state_list, baseline_aproximator, time_remaining, time_step, decay_term, verbose=False):
        # create tensors
        state_t = torch.FloatTensor(state_list).to(device)
        next_state_t = torch.FloatTensor(next_state_list).to(device)
        phi_t = torch.FloatTensor(selected_phi_list).to(device).view(-1,1)
        reward_t = torch.FloatTensor(reward_list).to(device).view(-1,1)
        
        # will need to transform rewards to returns? no we are boot strapping should we not bootstrap though... maybe who knows try both
        # boot strap a tunable baseline or only keep the last n actions?
        # also how to phase in the baseline slowly


        approx_value = baseline_aproximator.evaluate_prescribed(torch.linalg.norm(state_t, axis=1)).view(-1,1)
        sampled_q_val = reward_t + baseline_aproximator.evaluate_prescribed(torch.linalg.norm(next_state_t, axis=1)).view(-1,1)
        #approx_value = baseline_aproximator.mixed_evaluation(torch.linalg.norm(state_t, axis=1),time_remaining,decay_term).view(-1,1)
        #sampled_q_val = reward_t + baseline_aproximator.mixed_evaluation(torch.linalg.norm(next_state_t, axis=1),time_remaining+time_step,decay_term).view(-1,1)
        advantage_t = sampled_q_val - approx_value 
        #print(f"observed_separation {np.linalg.norm(state_list[-1])}")
        #print(f"advantage: {advantage_t[-1]}")

        

        outputs = self.actor_net(state_t)
        # testing phi is actually sampled
        distributions = torch.distributions.Normal(outputs[:,0], torch.exp(outputs[:,1]))
        log_probs = distributions.log_prob(phi_t)

        if verbose:#True:#verbose:
            print(f"observed_separation {np.linalg.norm(state_list[-1])}")
            print(f"actual phi {phi_t[-1]}")
            print(f"mean: {outputs[:,0][-1]}")
            #print(f"log prob: {log_probs[-1]}")
            print(f"advantage: {advantage_t[-1]}")
            print(f"stds: {torch.exp(outputs[:,1][-1])}")
            #print(f"state approx value: {approx_value[-1]}")
            #print(f"next state approx value: {bootstrap_value[-1]}")
            print()
        


        
        actor_loss = torch.mean(-log_probs * advantage_t)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
 #       torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1.)
        self.actor_optimizer.step() 



        return actor_loss.detach().cpu().numpy(), advantage_t.detach().cpu().numpy()
    



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



# train and save model and save reward/loss graphs in the abc environment
if __name__=="__main__":

    import matplotlib.pyplot as plt
    import time
    from abc_env_stoch import ABCflow


    for baseline_phi in range(5,8):
    #for baseline_phi in range(5,6):
        phi_aproximator = Baseline_reworked(baseline_phi/10,0.25187,0.27710,3.,0.02*np.pi) # values are taken from sampling the lyapunov exponenet DANGER HARDCODED

        agent = DynamicPhiAgent(action_scale=5.)
        env = ABCflow()
        #env = ABCflow(a=0.,b=0.,c=0.)
        env.deltaT= 0.05
        env.kappa = (0.02*np.pi)**2 * phi_aproximator.lyapunov
        env.limit=10.
        num_ep=100
        stats_actor_loss, rewards_per_episode = [], []
        verbose_episodes = 50
        episode_step_count = int(env.limit/env.deltaT)
        advantages = np.empty((0,episode_step_count),float)


        for ep in range(num_ep):
            rewards = []
    #        time.sleep(5)
            state = env.getState()
            state_np_arr, next_state_arr = np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
            phi_list, reward_list, done_list = [], [], []
            verbose = ((ep + 1) % verbose_episodes) == 0


            while not env.isOver():
                time_remaining = env.limit - env.time # maybe add time remaining to state vector
                decay_term = np.exp(-ep/10)
                action = agent.sample_action(state)
                phi = (action @ state) / (state @ state)
                reward = env.step(action)
                #reward = -phi
                #print(f"reward is {reward}")
                rewards.append(reward)
                next_state = env.getState()

                state_np_arr = np.vstack((state_np_arr,state))
                phi_list.append(phi)
                reward_list.append(reward)
                next_state_arr = np.vstack((next_state_arr,next_state))
                done_list.append(1. - env.isOver())



                actor_loss, ep_advantages = agent.train(state_np_arr, phi_list, reward_list, next_state_arr, phi_aproximator, time_remaining,env.deltaT, decay_term,verbose = verbose)
                stats_actor_loss.append(actor_loss)
                

                state = next_state
            #print(ep_advantages.T.shape)
            advantages = np.vstack((advantages,ep_advantages.T))
            rewards_per_episode.append(sum(rewards)/len(rewards))
            print(f"episode {ep} termination distance {env.dist()}")
            env.reset()


        np.savetxt(f"csv/baseline_phi_untuned={baseline_phi/10}.csv", advantages, delimiter=",")
        np.savetxt(f"csv/loss_phi_untuned={baseline_phi/10}.csv", stats_actor_loss, delimiter=",")
        np.savetxt(f"csv/rewards_phi_untuned={baseline_phi/10}.csv", rewards_per_episode, delimiter=",")


        window_size = 50
        avg_policy_loss = []

        # Calculate sliding window averages
        for i in range(len(stats_actor_loss) - window_size + 1):
            window = stats_actor_loss[i : i + window_size]
            window_avg = np.mean(window)
            avg_policy_loss.append(window_avg)




        plt.plot(rewards_per_episode)
        plt.savefig('actor_physics_reward_per_episode_untuned.png')
        plt.clf()
        plt.plot(avg_policy_loss)
        plt.savefig('actor_physics_loss_untuned.png')
        plt.clf()

        agent.save_policy(f"saved_models/phi_only_{baseline_phi/10}_untuned.pt")
