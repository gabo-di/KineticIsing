using Statistics
using NPZ
using Revise
includet("plefka.jl")



function nsf(num, n=4)
    """n-Significant Figures"""
    return round(num, digits=n-1)
end

function main()
    sze = 512     
    R = 1000000                # Repetitions of the simulation
    H0 = 0.5                   # Uniform distribution of fields parameter
    J0 = 1.0                   # Average value of couplings
    Js = 0.1                   # Standard deviation of couplings

    B = 21                    # Number of values of beta
    T = 2^7                  # Number of simulation time steps

    betas = collect( 1 .+ range(-1, 1, length=B) .* 0.3)
    for ib in 1:B 
        beta_ref = round(betas[ib], digits=3)

        EmP_t1_t = zeros(T + 1)
        EmP_t = zeros(T + 1)
        EmP2_t = zeros(T + 1)
        EmP_t1 = zeros(T + 1)

        ECP_t1_t = zeros(T + 1)
        ECP_t = zeros(T + 1)
        ECP2_t = zeros(T + 1)
        ECP_t1 = zeros(T + 1)

        EDP_t1_t = zeros(T + 1)
        EDP_t = zeros(T + 1)
        EDP2_t = zeros(T + 1)
        EDP_t1 = zeros(T + 1)

        mP_t1_t_mean = ones(T + 1)
        mP_t_mean = ones(T + 1)
        mP2_t_mean = ones(T + 1)
        mPexp_mean = ones(T + 1)
        mP_t1_mean = ones(T + 1)
        mP_t1_t_final = zeros(sze)
        mP_t_final = zeros(sze)
        mP2_t_final = zeros(sze)
        mP_t1_final = zeros(sze)
        mPexp_final = zeros(sze)

        CP_t1_t_mean = zeros(T + 1)
        CP_t_mean = zeros(T + 1)
        CP2_t_mean = zeros(T + 1)
        CPexp_mean = zeros(T + 1)
        CP_t1_mean = zeros(T + 1)
        CP_t1_t_final = zeros((sze, sze))
        CP_t_final = zeros((sze, sze))
        CP2_t_final = zeros((sze, sze))
        CP_t1_final = zeros((sze, sze))
        CPexp_final = zeros((sze, sze))

        DP_t1_t_mean = zeros(T + 1)
        DP_t_mean = zeros(T + 1)
        DP2_t_mean = zeros(T + 1)
        DPexp_mean = zeros(T + 1)
        DP_t1_mean = zeros(T + 1)
        DP_t1_t_final = zeros((sze, sze))
        DP_t_final = zeros((sze, sze))
        DP2_t_final = zeros((sze, sze))
        DP_t1_final = zeros((sze, sze))
        DPexp_final = zeros((sze, sze))

        # Load data
        filename = "data/data-H0-" * string(H0) * "-J0-" * string(J0) * "-Js-" * string(
            Js) * "-N-" * string(sze) * "-R-" * string(R) * "-beta-" * string(beta_ref) * ".npz"
        println(filename)
        data = npzread(filename)
        H = data["H"]
        J = data["J"]
        s0 = data["s0"]
        m_exp = data["m"]
        C_exp = data["C"]
        D_exp = data["D"]
        data = nothing

        # Load statistical moments from data
        mPexp_mean[1] = 1
        for t in 1:T
            mPexp_mean[t + 1] = mean(m_exp[:, t])
            CPexp_mean[t + 1] = mean(C_exp[:, :, t])
            DPexp_mean[t + 1] = mean(D_exp[:, :, t])
            # println("Exp",
            #       string(t) * "/" * string(T),
            #       mPexp_mean[t + 1],
            #       CPexp_mean[t + 1],
            #       DPexp_mean[t + 1])
        end
        mPexp_final = m_exp[:, T - 1]
        CPexp_final = C_exp[:, :, T - 1]
        DPexp_final = D_exp[:, :, T - 1]

        # run Plefka[t-1,t] order 2
        println("Plefka[t-1,t]")
        time_P_t1_t = calc_stuff!(CP_t1_t_mean, mP_t1_t_mean, DP_t1_t_mean,
                    EmP_t1_t, ECP_t1_t, EDP_t1_t,
                    m_exp, C_exp, D_exp,
                    mP_t1_t_final, CP_t1_t_final, DP_t1_t_final,
                    T, beta_ref, IsingPlefka_t1_t{Float64}(),
                    H, J, s0, sze)
        println(time_P_t1_t)

        # run Plefka[t] order 2
        println("Plefka[t]")
        time_P_t = calc_stuff!(CP_t_mean, mP_t_mean, DP_t_mean,
                    EmP_t, ECP_t, EDP_t,
                    m_exp, C_exp, D_exp,
                    mP_t_final, CP_t_final, DP_t_final,
                    T, beta_ref, IsingPlefka_t{Float64}(),
                    H, J, s0, sze)
        println(time_P_t)

        # run Plefka[t-1] order 1
        println("Plefka[t-1]")
        time_P_t1 = calc_stuff!(CP_t1_mean, mP_t1_mean, DP_t1_mean,
                    EmP_t1, ECP_t1, EDP_t1,
                    m_exp, C_exp, D_exp,
                    mP_t1_final, CP_t1_final, DP_t1_final,
                    T, beta_ref, IsingPlefka_t1{Float64}(),
                    H, J, s0, sze)
        println(time_P_t1)

        # run Plefka[t-1] order 2
        println("Plefka2[t]")
        time_P2_t = calc_stuff!(CP2_t_mean, mP2_t_mean, DP2_t_mean,
                    EmP2_t, ECP2_t, EDP2_t,
                    m_exp, C_exp, D_exp,
                    mP2_t_final, CP2_t_final, DP2_t_final,
                    T, beta_ref, IsingPlefka2_t{Float64}(),
                    H, J, s0, sze)
        println(time_P2_t)


    # Save results to file

    filename = "data/forward/forward_" * string(Int(
        round(beta_ref * 100, digits=0))) * "_R_" * string(R) * ".npz"

    npzwrite(filename,
                        m_exp=m_exp[:, T - 1],
                        C_exp=C_exp[:, :, T - 1],
                        D_exp=D_exp[:, :, T - 1],
                        mP_t1_t_mean=mP_t1_t_mean,
                        mP_t_mean=mP_t_mean,
                        mP_t1_mean=mP_t1_mean,
                        mP2_t_mean=mP2_t_mean,
                        CP_t1_t_mean=CP_t1_t_mean,
                        CP_t_mean=CP_t_mean,
                        CP_t1_mean=CP_t1_mean,
                        CP2_t_mean=CP2_t_mean,
                        DP_t1_t_mean=DP_t1_t_mean,
                        DP_t_mean=DP_t_mean,
                        DP_t1_mean=DP_t1_mean,
                        DP2_t_mean=DP2_t_mean,
                        mPexp_mean=mPexp_mean,
                        CPexp_mean=CPexp_mean,
                        DPexp_mean=DPexp_mean,
                        mP_t1_t=mP_t1_t_final,
                        mP_t=mP_t_final,
                        mP_t1=mP_t1_final,
                        mP2_t=mP2_t_final,
                        CP_t1_t=CP_t1_t_final,
                        CP_t=CP_t_final,
                        CP_t1=CP_t1_final,
                        CP2_t=CP2_t_final,
                        DP_t1_t=DP_t1_t_final,
                        DP_t=DP_t_final,
                        DP_t1=DP_t1_final,
                        DP2_t=DP2_t_final,
                        EmP_t1_t=EmP_t1_t,
                        EmP_t=EmP_t,
                        EmP_t1=EmP_t1,
                        EmP2_t=EmP2_t,
                        ECP_t1_t=ECP_t1_t,
                        ECP_t=ECP_t,
                        ECP_t1=ECP_t1,
                        ECP2_t=ECP2_t,
                        EDP_t1_t=EDP_t1_t,
                        EDP_t=EDP_t,
                        EDP_t1=EDP_t1,
                        EDP2_t=EDP2_t,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t=time_P_t,
                        time_P_t1=time_P_t1,
                        time_P2_t=time_P2_t)
        
    end
end

function calc_stuff!(CP_mean, mP_mean, DP_mean, 
                    EmP, ECP, EDP,
                    m_exp, C_exp, D_exp,
                    mP_final, CP_final, DP_final,
                    T, beta_ref, alg,
                    H, J, s0, sze)
    # Initialize kinetic Ising model
    tt = Float64
    I = Ising(tt, sze)
    I.H .= copy(H)
    I.J .= copy(J)

    MI = MeanIsingModel(tt, sze, alg)
    initialize_state!(MI, s0)


    # Run Plefka[t-1,t], order 2
    time_start = time()
    for t in 1:T
        update_P!(I, MI)
        CP_mean[t + 1] = mean(MI.C)
        mP_mean[t + 1] = mean(MI.m)
        DP_mean[t + 1] = mean(MI.D)
        EmP[t + 1] = mean((MI.m .- m_exp[:, t]).^2)
        ECP[t + 1] = mean((MI.C .- C_exp[:, :, t]).^2)
        EDP[t + 1] = mean((MI.D .- D_exp[:, :, t]).^2)
        # print("beta",
        #       beta_ref,
        #       "P_t1_t_o2",
        #       string(t) * "/" * string(T),
        #       nsf(mP_mean[t + 1]),
        #       nsf(CP_mean[t + 1]),
        #       nsf(DP_mean[t + 1]),
        #       nsf(EmP[t + 1]),
        #       nsf(ECP[t + 1]),
        #       nsf(EDP[t + 1]))
    end
    mP_final .= MI.m
    CP_final .= MI.C
    DP_final .= MI.D
    time_P = time() - time_start
    return time_P
end
