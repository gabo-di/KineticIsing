using NonlinearSolve
using LinearAlgebra
using LoopVectorization
#using Integrals
using SparseArrays
using Trapz
import Base.\

# this is in order to use SimpleNewtonRaphson(), see raphson.jl:50 
# also that wee need to use not inplace function so we not do a cache for the jacobian, see utils.jl:156
\(x::AbstractVector, y::AbstractVector) = y ./ x

function TAP_eq(x, H, Vii)
    tanh.(H .- x.*Vii) .- x
end

function diff_TAP_eq(x, H, Vii)
    # diagm(-Vii.*(1 .- tanh.(H .- x.*Vii).^2) .- 1)
    -Vii.*(1 .- tanh.(H .- x.*Vii).^2) .- 1
end

function solve_TAP_eq(x0, H, Vii, TOL=1e-15)
    f(x,p) = TAP_eq(x, p[1], p[2])
    df(x,p) = diff_TAP_eq(x, p[1], p[2])
    F = NonlinearFunction(f, jac=df)
    prob = NonlinearProblem(F, x0, [H, Vii]; abstol=TOL)
    sol = solve(prob,SimpleNewtonRaphson())
    sol.u
end

# PLEFKA[t-1,t] order 1
function update_m_P_t1_t_o1(H, J, m)
    tanh.(H + J*m)
end

function update_C_P_t1_t_o1(H, J, m)
    diagm(1 .- m.^2)
end

function update_D_P_t1_t_o1(H, J, m, m_p)
    (1 .- m.^2).*J.*(1 .- m_p.^2)'
end

# PLEFKA[t-1,t] order 2
function update_m_P_t1_t_o2(H, J, m_p) 
    Vii = (J.^2)*( 1 .- m_p.^2)
    Heff = H + J*m_p
    return solve_TAP_eq(m_p, Heff, Vii)
end

function update_C_P_t1_t_o2(H, J, m, m_p)
    x = (1 .- m.^2)
    C = x .* J*((1 .- m_p.^2) .* J') .* x'
    @turbo for i in eachindex(x)
        C[i,i] = x[i]
    end
    return C
end

function update_D_P_t1_t_o2(H, J, m, m_p)
    ((1 .- m.^2).*J.*(1 .- m_p.^2)') .* (1 .+ 2 .* m .* J .* m_p')
    # ojo aqui esta sin el 2, ver ecuacion 94 de supplementary material
end

# PLEFKA[t] order 1
function update_m_P_t_o1(H, J, m)
    tanh.(H + J*m)
end
    
function update_C_P_t_o1(H, J, m_p)
    diagm(1 .- m_p.^2)
end

function update_D_P_t_o1(H, J, m, C_p)
    (1 .- m.^2) .* J * C_p
end

# PLEFKA[t] order 2
function update_m_P_t_o2(H, J, m_p, C_p)
    # ein"ij,il,jl->i"(A,B,C) ≈ (A .* (B * C')) * ones(n)
    n = size(m_p,1)
    Vii = (J.* (J * C_p')) * ones(n)
    Heff = H + J*m_p
    solve_TAP_eq(m_p, Heff, Vii)
end

function update_C_P_t_o2(H, J, m, C_p)
    # ein"i,k,ij,kl,jl->ik"(x,y,A,B,C) ≈ ( x .* A * C * B' .* y')
    x = (1 .- m.^2)
    C = x .* J * C_p * J' .* x'
    @turbo for i in eachindex(x)
        C[i,i] = x[i]
    end
    return C
end

function update_D_P_t_o2(H, J, m, m_p, C_p)
    # ein"i,ij,jl->il"(x,A,B) ≈ ( x .* A * B )
    # ein"i,ij,il,jl,l->il"(x,A,B,C,y) ≈ x .* A * C .* B .* y'
    x = (1 .- m.^2)
    (x .* J * C_p) + (2 .* m .* x) .* J * C_p .* J .* m_p'
end

# PLEFKA[t-1] order 1
function integrate_1DGaussian(f, args)
    x = collect(range(-4, 4, length=20))
    y = f(x, args...) 
    # trapezoidal integration
    trapz(x, y)

    # rectangle integration
    # sum(y)*(x[2] - x[1])

    # Adaptative integration
    # prob = IntegralProblem((x,p)->f(x,p...), -4, 4, args)
    # sol = solve(prob, QuadGKJL(); reltol = 1e-8, abstol = 1e-8)
    # return sol.u[1]
end

function integrate_2DGaussian(f,args)
    x = collect(range(-4, 4, length=20))
    y = f(x, x', args...) 
    # trapezoidal integration
    trapz((x, x), y)

    # rectangle integration
    # sum(y)*(x[2] - x[1])^2

    # Adaptative integration
    # prob = IntegralProblem((x,p)->f(x[1],x[2],p...), [-4, -4], [4, 4], args)
    # sol = solve(prob, HCubatureJL(); reltol = 1e-8, abstol = 1e-8)
    # return sol.u[1]
end

function dT1(x, g, D)
    return @. 1/sqrt(2pi) * exp(-x^2/2) * tanh(g + x*sqrt(D))
end

function dT1_1(x, g, D)
    return @. 1/sqrt(2pi) * exp(-x^2/2) * (1 - tanh(g + x*sqrt(D))^2)
end

function dT1_2(x, g, D)
    return @. 1/sqrt(2pi) * exp(-x^2/2) * (-2*tanh(g + x*sqrt(D))) * (1 - tanh(g + x*sqrt(D))^2)
end

function update_m_P_t1_o1(H, J, m_p)
    m = zero(H)
    g = H + J*m_p
    D = J.^2 * (1 .- m_p.^2)
    for i in eachindex(m)
        m[i] = integrate_1DGaussian(dT1, (g[i], D[i])) 
    end
    return m
end

function update_D_P_t1_o1(H, J, m_p, C_p)
    a = zero(H)
    g = H + J*m_p
    D = J.^2 * (1 .- m_p.^2)
    for i in eachindex(a)
        a[i] = integrate_1DGaussian(dT1_1, (g[i], D[i])) 
    end
    # ein"i,ij,jl->il"(x,A,B) ≈ x .* A * B
    a .* J * C_p
end

function dT2_rot(p, gx, gy, Dx, Dy, rho)
    return @. 1/sqrt(2pi) * exp(-p^2/2) * tanh(gx + p*sqrt(1+rho)*sqrt(Dx/2)) *
    tanh(gy + p*sqrt(1+rho)*sqrt(Dy/2)) 
end

function dT2_rot(p, n, gx, gy, Dx, Dy, rho)
    return @. 1/(2pi) * exp(-(p^2 + n^2)/2) *
    tanh(gx + (p*sqrt(1+rho) + n*sqrt(1-rho))*sqrt(Dx/2)) *
    tanh(gy + (p*sqrt(1+rho) - n*sqrt(1-rho))*sqrt(Dy/2)) 
end

function update_C_P_t1_o1(H, J, m, m_p, C_p)
    n = length(m)
    C = zero(J)
    g = H + J*m_p
    D = J.^2 * (1 .- m_p.^2)
    inv_D = zero(D)
    inv_D[D .> 0 ] = 1 ./ D[ D.> 0]
    #ein"i,k,ij,kj,j->ik"(x,y,A,B,z) ≈ x .* A * (z .* B') .* y'
    rho = sqrt.(inv_D) .* J * ((1 .- m_p.^2) .* J' ) .* sqrt.(inv_D)'
    # for i in eachindex(m)
    Threads.@threads for i in eachindex(m)
        C[i,i] = 1 - m[i]^2
        for j in (i+1:n)
            if rho[i,j] > (1 - 1e-5)
                # ojo en el original usa 1 - 1e5, pero eso no tiene sentido, la idea es evitar que 1-rho^2 < 0
                C[i,j] = integrate_1DGaussian(dT2_rot, (g[i], g[j], D[i], D[j], rho[i,j])) - m[i]*m[j]
            else
                C[i,j] = integrate_2DGaussian(dT2_rot, (g[i], g[j], D[i], D[j], rho[i,j])) - m[i]*m[j]
            end
            C[j,i] = C[i,j]            
        end
    end
    return C
end

#PLEFKA2[t] order 2

function TAP_eq_D(x, Heff, V)
    x .- Heff .+ tanh.(x) .* V
end

function TAP_eq_D_v1(du, u, p)
    du .= u .- p[1] .+ tanh.(u) .* p[2]
    nothing
end

function diff_TAP_eq_D(x, Heff, V)
    # Diagonal(1 .+ (1 .- tanh.(x).^2) .* V)
    1 .+ (1 .- tanh.(x).^2) .* V
end

function diff_TAP_eq_D_v1(du, u, p)
    du .= spdiagm( 1 .+ (1 .- tanh.(u).^2) .* p[2])
    nothing
end

function solve_TAP_eq_D_v0(x0, Heff, V, TOL=1e-15)
    x = deepcopy(x0)
    tap = TAP_eq_D(x, Heff, V)
    # dtap = diff_TAP_eq_D(x, Heff, V)
    # z = abs.(tap) .> TOL
    error = maximum(abs.(tap))
    while error>TOL
        tap = TAP_eq_D(x, Heff, V)
        z = abs.(tap) .> TOL
        dtap = diff_TAP_eq_D(x, Heff, V)
        x[z] .-= tap[z] ./ dtap[z]
        error = maximum(abs.(tap))
    end
    return x
end

function solve_TAP_eq_D(x0, Heff, V, TOL=1e-15)
    s = size(x0)
    f(x,p) = TAP_eq_D(x, p[1], p[2])
    df(x,p) = diff_TAP_eq_D(x, p[1], p[2])
    F = NonlinearFunction(f, jac=df)
    prob = NonlinearProblem(F, reshape(x0, :), [reshape(Heff, :), reshape(V, :)]; abstol=TOL)
    sol = solve(prob,SimpleNewtonRaphson()) # for small problems
    # sol = solve(prob, NewtonRaphson())
    reshape(sol.u, s)
end

function solve_TAP_eq_D_v1(x0, Heff, V, TOL=1e-15)
    s = size(x0)
    # F = NonlinearFunction(TAP_eq_D_v1, jac=diff_TAP_eq_D_v1) # for small problems
    F = NonlinearFunction(TAP_eq_D_v1,  sparsity=spdiagm(reshape(x0,:))) # for big problems 
    prob = NonlinearProblem(F, reshape(x0,:), [reshape(Heff,:) , reshape(V,:)]; abstol=TOL)
    # sol = solve(prob, SimpleNewtonRaphson())
    sol = solve(prob, NewtonRaphson())
    reshape(sol.u, s)
end

function update_D_P2_t_o2(H, J, m_p, C_p, D_p)
    n = length(H)
    D = zero(J)
    m_D = zero(H)
    o = ones(n,n)

    Heff = H + J*m_p
    Heff_i = Heff .* o 
    # ein"ij,in,jn->i"(A,B,C) ≈ (A .* (B * C')) * ones(n)
    V_p = (J .* (J * C_p')) * ones(n) 
    # ein"ij,ln,jn->il"(A,B,C) ≈ (A * C * B')
    W_p = J * D_p * J' 

    m_i = zero(J)
    m_pil = o .* m_p'
    V_pil = V_p .* o
    # ein"il,in,ln->il"(A,B,C) ≈  A .* (B * C')
    V_pil -= 2* J.* (J * C_p')
    V_pil += J.^2 * Diagonal(C_p)
    # ein"il,ln,ln->il"(A,B,C) ≈ A .* ((B .* C) * ones(n))'
    W_pi1 = W_p - J .* ( (J .* D_p) * ones(n))'

    Delta_il = J + W_pi1

    for sl in [-1;1]
        Heff_il = Heff_i .+ Delta_il .* (sl .- m_pil)
        theta = solve_TAP_eq_D(Heff_il, Heff_il, V_pil)
        D += tanh.(theta) .* sl .* (1 .+ sl * m_pil ) ./ 2
        m_i += tanh.(theta) .* (1 .+ sl * m_pil) ./ 2
    end

    D -= m_i .* m_pil
    m_D = m_i * ones(n)/n
    return m_D, D
end

function update_C_P2_t_o2(H, J, m, m_p, C_p)
    n = length(H)
    C_D = zero(J)
    o = ones(n,n)

    Heff= H + J*m_p
    # ein"ij,il,jl->i"(A,B,C) ≈ (A .* (B * C')) * ones(n)
    V_p = (J.* (J * C_p')) * ones(n)
    V_pik = V_p .* o
    # ein"ij,kl,jl->ik"(A,B,C) ≈ A * C * B'
    W_p = J * C_p * J' 

    m_i = zero(J)
    m_pik = o .* m_p' 
    Heff_i = Heff .* o

    Delta_ik = W_p

    for sk in [-1,1]
        Heff_ik = Heff_i + Delta_ik .* (sk .- m_pik)
        theta = solve_TAP_eq_D(Heff_ik, Heff_ik, V_pik)
        C_D += tanh.(theta) .* (sk .- m_pik) .* (1 .+ sk .* m_pik) ./ 2
    end
    C_D = (C_D + C_D') / 2

    @turbo for i in eachindex(m)
        C_D[i,i] = 1 - m[i]^2
    end
    return C_D
end


