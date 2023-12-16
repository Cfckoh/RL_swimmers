import numpy as np
import sdeint
from itertools import accumulate





def control_2d(u, t, *args):
    control = args[0]
    return np.array([
        u[2]*u[0] + u[3]*u[1] - control[0],
        u[4]*u[0] + u[5]*u[1] - control[1],
        -u[2],
        -u[3],
        -u[4],
        -u[5]
    ])

def stoch_flow_2d(u, t, *args):

    kappa = args[0]
    D = args[1]
    return np.array([
        [np.sqrt(kappa), 0, 0, 0, 0],
        [0, np.sqrt(kappa), 0, 0, 0],
        [0, 0, np.sqrt(D), 0, 0],
        [0, 0, 0, np.sqrt(D), np.sqrt(2) * np.sqrt(D)],
        [0, 0, 0, np.sqrt(D), -np.sqrt(2) * np.sqrt(D)],
        [0, 0, -np.sqrt(D), 0, 0]
    ])



def closure(func, *args):
    def newfunc(x, t):
        return func(x, t, *args)
    return newfunc




class synthetic_env():
    
    def __init__(self,start_sep=0.001, kappa=0.02*np.pi, D=0.25,seed=1):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.start_sep = start_sep
        self.sep_vec = self.rng.random(2) - 0.5
        self.sep_vec = self.sep_vec*start_sep/np.linalg.norm(self.sep_vec)
        # uniform random generatio anywhere in a two pi periodic space
        self.reward=0
        self.deltaT=0.01
        self.time=0
        self.limit=10.
        self.kappa = kappa
        self.D = D
        self.sigmas = np.zeros(4)#self.rng.normal(scale=np.sqrt(self.deltaT), size=(3))
        
    def reset(self):
        self.sep_vec = self.rng.random(2) - 0.5
        self.sep_vec = self.sep_vec*self.start_sep/np.linalg.norm(self.sep_vec)
        self.reward=0
        self.time=0
        self.sigmas = np.zeros(4)#self.rng.normal(scale=np.sqrt(self.deltaT), size=(3))

    
    def step(self,action):
        action=action
        state = np.hstack((self.sep_vec,self.sigmas))
        new_state = sdeint.itoint(closure(control_2d,action),
                                     closure(stoch_flow_2d, self.kappa , self.D),
                                     state,
                                     [self.time,self.time+self.deltaT],
                                     generator = self.rng
                                    )[-1]
        self.sep_vec = new_state[0:2]
        self.sigmas = new_state[2:]
        self.time+=self.deltaT
        self.reward=-(self.dist()**2+action@action)*self.deltaT
        return self.reward
    
    #state is just the seperation vector
    def getState(self):
        return self.sep_vec
    
    def isOver(self):
        return self.time >= self.limit
        
    def dist(self):
        return np.sqrt(self.sep_vec @ self.sep_vec)
   
    def create_cov_mat(self, dim,D):
        cov_mat = np.zeros((dim**2,dim**2))
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        term1 = (j==l)*(i==k)
                        term2 = (i==j)*(k==l)
                        term3 = (j==k)*(i==l)
                        val = term1 - (term2+term3)/(dim+1)
                        index1 = dim*i + j
                        index2 = dim*l + k
                        cov_mat [index1,index2] = val

        cov_mat = D*(dim+1)*cov_mat

        return cov_mat