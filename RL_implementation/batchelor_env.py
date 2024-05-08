from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Use if using a julia image this is faster see: https://pyjulia.readthedocs.io/en/latest/troubleshooting.html 
#import julia
#jl = julia.Julia(sysimage="julia_img.so")
#from julia import Main

import numpy as np


class BatchelorFlows():
    """
    Environment is 2D synthetic environment!! 
    """
    def __init__(self,start_sep, beta, kappa, D, nu,seed=1):
        Main.include("batchelor_numerics.jl")
        self.rng = np.random.default_rng(seed=seed)
        self.start_sep = start_sep
        self.sep_vec = self.rng.random(2) - 0.5
        self.sep_vec = self.sep_vec*start_sep/np.linalg.norm(self.sep_vec)
        self.reward=0
        self.deltaT=0.1 # environment step size
        self.time_step=0
        self.limit=10.
        self.kappa = kappa
        self.D = D
        self.sigmas = np.zeros(4)
        self.nu = nu
        self.beta = beta

    def reset(self):
        self.sep_vec = self.rng.random(2) - 0.5
        self.sep_vec = self.sep_vec*self.start_sep/np.linalg.norm(self.sep_vec)
        self.reward=0
        self.time_step=0
        #don't reset sigmas we want colored noise generators to start as non-zero
        #self.sigmas = np.zeros(4)  

    def step(self,action):
        # actions are just a selection of phi for this
        phi = action
        state = np.hstack((self.sep_vec,self.sigmas))
        loc_string = np.array2string(state, separator=",")
        new_state, penalty = Main.eval(f"envStep({self.kappa}, {self.beta}, {self.D}, {phi}, {self.nu},{loc_string},{self.deltaT})") 
        self.sep_vec = new_state[0:2]
        self.sigmas = new_state[2:]
        self.time_step+=1
        reward = -penalty
        return reward


    def eval_step(self,phi):
        """
        Returns the baseline aproximate for the given phi value and the current state

        NOTE: In training you should always evaluate for a fixed phi not the phi the agent picks. 

        """

        dims = 2
        d_tilde = 4 * self.D
        #useful intermediate value used in the baseline mutiple times
        block = self.nu + 2*phi - d_tilde
        time_remaining = self.limit - self.getTime()
        a = ((self.beta + phi**2)*(1-np.exp(-time_remaining*block)))/block
        b_term1 = dims * self.kappa * (self.beta + phi**2) / (self.nu*(2*phi-d_tilde))
        b_term2 = 1 - np.exp(-self.nu*time_remaining) - self.nu * (1-np.exp(-time_remaining*block))/block
        b = b_term1 * b_term2
        return -(a*self.dist()**2 + b)




    def getState(self):
        return self.sep_vec
    
    def isOver(self):
        return self.time_step * self.deltaT >= self.limit
        
    def dist(self):
        return np.linalg.norm(self.sep_vec)

    def getTime(self):
        return self.time_step * self.deltaT


