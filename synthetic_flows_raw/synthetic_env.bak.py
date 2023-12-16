import numpy as np
import sdeint
from itertools import accumulate

#\"""

# TIME CORRELATED

def control_2d(x, t, *args):
    "Deterministic part of the flow handles time correlation and control"
    control = args[0]
    
    #tao = args[1]
    #r_1 = x[0]
    #r_2 = x[1]
    #sigma_11 = x[2]
    #sigma_12 = x[3]
    #sigma_21 = x[4]
    #sigma_22 = x[5]
    #dx=np.zeros(6)
    #dx[0] = sigma_11*r_1*sigma_12*r_2-control[0]
    #dx[1] = sigma_21*r_1*sigma_22*r_2-control[1]
    #dx[2:]=x[2:]*(-1/tao)   
    #print(x)

    dx= np.array([
        x[2]*x[0] + x[3]*x[1] - control[0],
        x[4]*x[0] + x[5]*x[1] - control[1],
        -x[2],
        -x[3],
        -x[4],
        -x[5]
    ])


    return dx

def stoch_flow_2d(x, t, *args):
    """
    The stochastic flows returns an array that is multiplied by the wiener proccesses
    Array is mostly diagonal?
    """
    #print(x)
    kappa = args[0]
    D = args[1]
    arr = np.zeros((6,5))
    arr[0,0] = arr[1,1] = np.sqrt(kappa)
    arr[2,2] = np.sqrt(D)
    arr[3,3] = np.sqrt(D)
    arr[3,4] = np.sqrt(2*D)
    arr[4,3] = np.sqrt(D)
    arr[4,4] = -np.sqrt(2*D)
    arr[5,2] = -np.sqrt(D)
    return arr
#\"""
"""
# No time correlation

def control_2d(x, t, *args):
    dx=np.zeros(0)

    return dx

def stoch_flow_2d(x, t, *args):
    arr = np.zeros(2,5)

    return arr
"""

def closure(func, *args):
    def newfunc(x, t):
        return func(x, t, *args)
    return newfunc



"""
TODO

then test against og environment a bit

NOTE Need to find propper kappa noise value

"""


class synthetic_env():
    
    def __init__(self,start_sep=0.2*np.pi, kappa=0.01, D=.1,time_cor=0.5,seed=1):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.start_sep = start_sep
        #self.sep_vec = self.rng.random(3) - 0.5
        self.sep_vec = np.zeros(2)#self.sep_vec*start_sep/np.linalg.norm(self.sep_vec)
        # uniform random generatio anywhere in a two pi periodic space
        self.reward=0
        self.deltaT=0.01
        self.time=0
        self.limit=10.
        self.kappa = kappa
        self.D = D
        self.time_cor = time_cor
        self.sigmas = np.zeros(4)#self.rng.normal(scale=np.sqrt(self.deltaT), size=(3))
        
    def reset(self):
        #self.sep_vec = self.rng.random(3) - 0.5
        self.sep_vec = np.zeros(2)#self.sep_vec*self.start_sep/np.linalg.norm(self.sep_vec)
        self.reward=0
        self.time=0
        self.sigmas = np.zeros(4)#self.rng.normal(scale=np.sqrt(self.deltaT), size=(3))

    
    def step(self,action):
        action=action
        state = np.hstack((self.sep_vec,self.sigmas))
        new_state = sdeint.itoint(closure(control_2d,action, self.time_cor),
                                     closure(stoch_flow_2d, np.sqrt(self.kappa) , self.D),
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