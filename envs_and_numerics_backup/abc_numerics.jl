using DifferentialEquations
using LinearAlgebra
using Random


function f(u,p,t)
    (A,B,C,PHI,kappa) = p
    swim_vec = u[1:3]-u[4:6]
    return [
        A*sin(u[3]) + C*cos(u[2])
        B*sin(u[1]) + A*cos(u[3])
        C*sin(u[2]) + B*cos(u[1])
        A*sin(u[6]) + C*cos(u[5]) + PHI * swim_vec[1]
        B*sin(u[4]) + A*cos(u[6]) + PHI * swim_vec[2]
        C*sin(u[5]) + B*cos(u[4]) + PHI * swim_vec[3]
    ]

end


function g(u,p,t)
    (A,B,C,PHI,kappa) = p
    return [ 
    sqrt(kappa) 0 0 0 0 0 
    0 sqrt(kappa) 0 0 0 0
    0 0 sqrt(kappa) 0 0 0
    0 0 0 sqrt(kappa) 0 0
    0 0 0 0 sqrt(kappa) 0
    0 0 0 0 0 sqrt(kappa)
    ]

end


function penalty(sep,dt,PHI,BETA)
    return (PHI^2+BETA)*sep^2*dt
end


function envStep(A, B, C, phi, NU, kappa, BETA, u0, delta_t)
    """
    Updates the current particle location after a delta_t sized environment step (not to be confused with the solver steps).
    Returns the new location of the particle and the penalty accrued over the step. 
    
    params: u0 is the a 6-tuple the first 3 coordinates coorsponding to the passive particle the second 3 to the active
    
    """
    dt = 0.005
    tspan = (0.0, delta_t)
    params = (A, B, C, phi, kappa)
    prob = SDEProblem(f, g, u0, tspan,params,noise_rate_prototype = zeros(6,6))


    sol = solve(prob,adaptive=false,dt=dt)
    trajs = reduce(hcat, sol.u)
    passive_traj = trajs[1:3,:]
    active_traj = trajs[4:6,:]

    sep_vecs = passive_traj - active_traj

    #return sep_vecs

    separations = sum(x -> x^2, sep_vecs; dims=1)
    separations .= sqrt.(separations)

    
    final_locs = trajs
    #calc discount and from that return
    discount = exp(-NU*dt)
    cost = 0.0
    # 2 to end so that we don't go off the end (RH Euler)
    
    
    for sep in reverse(separations[2:end])
        cost = cost * discount + penalty(sep,dt, phi, BETA)
    end

    a_end = active_traj[:,end]
    p_end = passive_traj[:,end]

    return p_end,a_end, cost
end
