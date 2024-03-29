import julia
jl = julia.Julia(sysimage="julia_img.so")
from julia import Main

import numpy as np


class ABC_env():
    """
    Environment is 3D ABC environment!! 
    """
    def __init__(self,A,B,C,start_sep, beta, kappa, D, nu,seed=1):
        """ 
        - proper value for D should be derived from sampled lyapunov exponent
        """
        Main.include("abc_numerics.jl")

        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.rng = np.random.default_rng(seed=seed)
        self.start_sep = start_sep
        self.sep_vec = self.rng.random(3) - 0.5
        self.sep_vec = self.sep_vec*start_sep/np.linalg.norm(self.sep_vec)
        self.passive = (self.rng.random(3) - 0.5) * 2 * np.pi
        self.active = self.active = self.passive + self.sep_vec
        self.reward=0
        self.deltaT=0.1 # environment step size
        self.time_step=0
        self.limit=10.
        self.kappa = kappa
        self.D = D
        self.nu = nu
        self.beta = beta

    # need to track active and passive not just separation vector

    def reset(self):
        self.sep_vec = self.rng.random(3) - 0.5
        self.sep_vec = self.sep_vec*self.start_sep/np.linalg.norm(self.sep_vec)
        self.passive = (self.rng.random(3) - 0.5) * 2 * np.pi
        self.active = self.active = self.passive + self.sep_vec
        self.reward=0
        self.time_step=0

    def step(self,action):
        # actions are just a selection of phi for this
        phi = action
        state_string = np.array2string(np.append(self.passive,self.active), separator=",")
        self.passive,self.active, penalty = Main.eval(f"envStep({self.A},{self.B},{self.C},{phi}, {self.nu}, {self.kappa}, {self.beta}, {state_string},{self.deltaT})") 
        self.sep_vec = self.passive - self.active
        self.time_step += 1
        reward = -penalty
        return reward


    def eval_step(self,phi):
        """
        Returns the baseline aproximate for the given phi value and the current state

        NOTE: In training you should always evaluate for a fixed phi not the phi the agent picks. 

        """

        dims = 3
        d_tilde = (dims+2) * (dims-1) * self.D
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
