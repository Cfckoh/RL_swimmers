
import csv
import numpy as np
import sdeint
import yaml
import sys

def f(u, t):
    swim_vector = u[0:3] - u[3:]
    return np.array([
        A*np.sin(u[2]) + C*np.cos(u[1]),
        B*np.sin(u[0]) + A*np.cos(u[2]),
        C*np.sin(u[1]) + B*np.cos(u[0]),
        A*np.sin(u[5]) + C*np.cos(u[4]) + PHI * swim_vector[0],
        B*np.sin(u[3]) + A*np.cos(u[5]) + PHI * swim_vector[1],
        C*np.sin(u[4]) + B*np.cos(u[3]) + PHI * swim_vector[2],
    ])

def g(u, t):
    return np.array([
        [np.sqrt(kappa), 0, 0, 0, 0, 0],
        [0, np.sqrt(kappa), 0, 0, 0, 0],
        [0, 0, np.sqrt(kappa), 0, 0, 0],
        [0, 0, 0, np.sqrt(kappa), 0, 0],
        [0, 0, 0, 0, np.sqrt(kappa), 0],
        [0, 0, 0, 0, 0, np.sqrt(kappa)],
    ])



def penalty(sep,phi,beta,delta_t):
    return (phi**2+beta)*sep**2*delta_t




if __name__=="__main__":
    
    # read config file passed in commandline
    config_file = f"{sys.argv[1]}"

    with open(f"baseline_config_files/{config_file}.yaml") as cf_file:
        config = yaml.safe_load( cf_file.read() )

    # load params from file
    dims=3 #abc flows is 3d
    A = config["A"]
    B = config["B"]
    C = config["C"]
    NU = config["NU"] 
    kappa = config["kappa"] 
    BETA = config["BETA"]
    PHI = config["PHI"]
    D = config["D"]
    delta_t = config["delta_t"]
    delta_r = config["delta_r"]
    t_end = config["t_end"]
    num_eps = config["num_eps"]


    results_dict = {}
    u0 = np.random.rand(6)-0.5
    for i in range(num_eps):


        tspan = np.arange(0.0, t_end,delta_t)  # 10 time units

        result = sdeint.itoint(f, g, u0, tspan)
        
        # storing last value for the random start of the next episode
        u0 = result[-1]

        sep_vecs = result[:,0:3]-result[:,3:]
        separations = np.sum(sep_vecs**2,axis=1)
        separations = np.sqrt(separations)


        N = len(separations)
        S_n = 0.0
        returns = np.zeros(N)
        discount = np.exp(-NU*delta_t)
        for i in range(1,N+1):
            S_n = S_n*discount + penalty(separations[N-i],PHI,BETA,delta_t)
            returns[N-i] = S_n
            
        
        for i in range(N):
            key = (i, int(separations[i]/delta_r))
            if key in results_dict:
                # increase sum and count
                results_dict[key] = results_dict[key][0] + returns[i], results_dict[key][1] + 1
            else:
                results_dict[key] = [returns[i], 1]




    with open(f'baseline_CSVs/{config_file}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results_dict.keys()))
        writer.writeheader()
        writer.writerow(results_dict)