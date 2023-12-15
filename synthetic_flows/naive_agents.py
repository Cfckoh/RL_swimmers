"""
A collection of agents which do not do any learning.
"""

import numpy as np


class DoNothing:
    """
    This agent always does nothing returns an array of zeros
    """
    def __init__(self):
        pass
        
    def sample_action(self,state):
        return np.zeros(3)


class FixedPhi():
    """
    Swims directly towards the passive particle with a fixed phi value 
    """
    def __init__(self, phi_value):
        self.phi=phi_value

    def sample_action(self, state):
        return state*self.phi

