import julia
jl = julia.Julia(sysimage="julia_img.so")#julia.Julia(compiled_modules=False)
from julia import Main

import numpy as np

class Test_Solver:
    def __init__(self):
        Main.include("juliaScript.jl")
    def solve_flow(self,phi):
        u0=np.zeros(6)
        loc = np.array2string(u0, separator=",")
        return Main.eval(f"envStep(0.001, 0.1, 0.1, {phi}, 0.99,{loc},1.)")
