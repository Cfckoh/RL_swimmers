using DifferentialEquations
using LinearAlgebra
using Random



function f(u,p,t)
    # r1 = u1, r2 = u2, s11=u3, s12=u4 s21=u5, s22=u6
    (kappa, BETA, D, PHI, NU) = p
    return [
        u[3]*u[1] + u[4]*u[2] - PHI * u[1]
        u[5]*u[1] + u[6]*u[2] - PHI * u[2]
        - u[3]
        - u[4]
        - u[5]
        - u[6]
    ]
end


function g(u,p,t)
    (kappa, BETA, D, PHI, NU) = p
    [
        sqrt(kappa) 0 0 0 0
        0 sqrt(kappa) 0 0 0
        0 0 sqrt(D) 0 0 
        0 0 0 sqrt(D) sqrt(2)*sqrt(D)
        0 0 0 sqrt(D) -sqrt(2)*sqrt(D)
        0 0 -sqrt(D) 0 0
    ]
end

function penalty(sep,delta_t,PHI,BETA)
    return (PHI^2+BETA)*sep^2*delta_t
end


function envStep(kappa, BETA, D, phi, NU, u0,delta_t)
    """
    Updates the current particle location after a delta_t sized environment step (not to be confused with the solver steps).
    Returns the new location of the particle and the penalty accrued over the step. 
    """
    dt = 0.005
    tspan = (0.0, delta_t)
    params = (kappa, BETA, D, phi, NU)
    prob = SDEProblem(f, g, u0, tspan,params,noise_rate_prototype = zeros(6, 5))
    #Random.seed!(2000)
    res = solve(prob,adaptive=false,dt=dt)
    u = reduce(hcat, res.u)
    u_end = u[:,end]
    
    separations = sum(x -> x^2, u[1:2,:]; dims=1)
    separations .= sqrt.(separations)
    
    #calc discount and from that return
    discount = exp(-NU*dt)
    cost = 0.0
    # 2 to end so that we don't go off the end (RH Euler)
    for sep in reverse(separations[2:end])
        cost = cost * discount + penalty(sep,dt, phi, BETA)
    end
    
    
    return u_end,cost
end
