import numpy as np
import yaml
import sys 
import matplotlib.pyplot as plt
from batchelor_env import BatchelorFlows
from actor_physicist_agent import RL_phi_agent

if __name__=="__main__":

    # get name of config file passed in commandline (without the .yaml extension)
    config_file = f"{sys.argv[1]}"

    with open(f"config_files/{config_file}.yaml") as cf_file:
        config = yaml.safe_load( cf_file.read() )

    # load params from file
    dims=2 #batchelor flow implementation is 2d
    start_sep = config["start_sep"]
    BETA = config["BETA"]
    kappa = config["kappa"] 
    D = config["D"]
    NU = config["NU"] 
    delta_t = config["delta_t"]
    t_end = config["t_end"]
    num_eps = config["num_eps"]
    baseline_phi = config["PHI"]
    verbose_episodes = config["verbose_episodes"]


    agent = RL_phi_agent(dims)
    env = BatchelorFlows(start_sep=start_sep,beta=BETA,kappa=kappa,D=D,nu=NU)
    env.deltaT = delta_t
    env.limit = t_end
    stats_actor_loss, rewards_per_episode = [], []


    #run episodes
    for ep in range(num_eps):
        #first intialize history data structs these are appened to every step
        state_np_arr, next_state_arr = np.array([]).reshape(0, dims), np.array([]).reshape(0, dims)
        state_val, next_state_val = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)
        action_list, reward_list = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)
        
        verbose = ((ep + 1) % verbose_episodes) == 0
        state = env.getState()

        while not env.isOver():
            action = agent.sample_action(state)
            curr_val = env.eval_step(baseline_phi)
            reward = env.step(action)
            new_val = env.eval_step(baseline_phi)
            
            # append to history arrays
            next_state = env.getState()
            state_np_arr = np.vstack((state_np_arr,state))
            action_list = np.vstack((action_list,action))
            reward_list = np.vstack((reward_list,reward))
            next_state_arr = np.vstack((next_state_arr,next_state))
            state_val = np.vstack((state_val,curr_val))
            next_state_val = np.vstack((next_state_val,new_val))


            # make a training step and reocrd the loss
            actor_loss = agent.train(state_np_arr, action_list, reward_list, next_state_arr,state_val, next_state_val, verbose)
            stats_actor_loss.append(actor_loss)
            state = next_state

        rewards_per_episode.append(sum(reward_list))
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
    plt.savefig(f'training_graphs/rewards_{config_file}.png')
    plt.clf()

    # Loss is quite noisy so better to plot loss over an averaged window to see if we are headed in the right direction
    plt.plot(avg_policy_loss)
    plt.savefig(f'training_graphs/losses_{config_file}.png')
    plt.xlabel("# of steps")
    plt.ylabel("Reward")
    plt.clf()

    agent.save_policy(f"saved_models/{config_file}.pt")