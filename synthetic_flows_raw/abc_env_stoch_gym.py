import numpy as np
import sdeint
import gym
import numpy as np
from gym import spaces


def abc_flow(y, t, *args):
    """
    abc flow with added control velocity
    
    args:
    y (np.array): np.array of length 3 specifying a particles current location
    t (list): list of floats with the time steps to solve for
    a,b,c (float): coeffcients or the abc flows
    control (array): float array of length 3 with the velocity components of the control term
    
    """
    control=args[0]
    a=args[1]
    b=args[2]
    c=args[3]
    
    X, Y, Z = y #/10
    #dydt = 10*np.array([a*np.sin(Z)+c*np.cos(Y)+control[0], b*np.sin(X)+a*np.cos(Z)+control[1], c*np.sin(Y)+b*np.cos(X)+control[2]])
    dydt = np.array([a*np.sin(Z)+c*np.cos(Y)+control[0], b*np.sin(X)+a*np.cos(Z)+control[1], c*np.sin(Y)+b*np.cos(X)+control[2]])
    return dydt



def brownian(x, t, *args):
    var = args[0]
    arr = np.eye(3) * var
    return arr

def closure(func, *args):
    def newfunc(x, t):
        return func(x, t, *args)
    return newfunc



"""
TODO

then test against og environment a bit

NOTE Need to find propper kappa noise value

"""


class ABCflow_gym(gym.Env):
    
    def __init__(self,sep_size=0.02*np.pi, a=1.,b=.7,c=.43,seed=1,kappa=0.1):
        super().__init__()
        self.sep_size=sep_size
        self.rng = np.random.default_rng(seed=seed)
        # uniform random generatio anywhere in a two pi periodic space
        self.passive = self.rng.random(3) * 2 * np.pi
        self.active = self.active_start_loc()
        self.reward=0
        self.deltaT=0.01
        self.time=0
        self.limit=10.
        self.kappa = kappa
        self.a=a
        self.b=b
        self.c=c
        self.action_space = spaces.Box(low=np.array([-5.0]), high=np.array([5.0]))
        self.observation_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0]), high=np.array([1.0,1.0,1.0]))
        
        
        
    def reset(self):
        self.passive = self.rng.random(3) - 0.5
        self.active = self.active_start_loc()
        self.reward=0
        self.time=0
        info={}
        #print(self.getState().shape)
        return self.getState()#, info

    
    def step(self,phi_action):
        action=phi_action * self.getState()
        self.passive = sdeint.itoint(closure(abc_flow,np.zeros(3),self.a,self.b,self.c),
                                     closure(brownian, self.kappa*self.deltaT),
                                     self.passive,
                                     [self.time,self.time+self.deltaT])[-1]
        
        self.active = sdeint.itoint(closure(abc_flow,action,self.a,self.b,self.c),
                                     closure(brownian, self.kappa*self.deltaT),
                                     self.active,
                                     [self.time,self.time+self.deltaT])[-1]

        self.time+=self.deltaT
        self.reward=-(self.dist()**2+action@action)*self.deltaT
        
        info={}
        
        return self.getState(), self.reward, self.isOver(), {}
    
    #state is just the seperation vector
    def getState(self):
        return np.array(self.passive-self.active,np.float32)
    
    def isOver(self):
        return self.time >= self.limit
        
    def dist(self):
        seperation=self.passive-self.active
        return np.sqrt(seperation @ seperation)
   
    def active_start_loc(self):
        sep_vec = self.rng.random(3) - 0.5
        sep_vec = sep_vec*self.sep_size/np.linalg.norm(sep_vec)
        return self.passive + sep_vec
