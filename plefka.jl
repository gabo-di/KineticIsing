using ComponentArrays
using LinearAlgebra
using Random

include("plefka_functions.jl")

"""
this code is an implementation of paper 

[1]  Aguilera, M, Moosavi, SA & Shimazaki H (2021).
     A unifying framework for mean-field theories of asymmetric kinetic Ising systems.
     Nature Communications 12:1197; https://doi.org/10.1038/s41467-021-20890.
"""
############################
###  Kinetic Ising Model ###
############################

abstract type AbstractIsingModel{T} end
abstract type AbstractIsingCache{T} end


struct Ising{T} <: AbstractIsingModel{T} 
    sze::Int
    H::AbstractArray{T,1}
    J::AbstractArray{T,2}
    Beta::T
    s::AbstractArray{T,1}
end

struct IsingCache_v0{T} <: AbstractIsingCache{T}
    h::AbstractArray{T,1}
    r::AbstractArray{T,1}
    z::AbstractArray{Bool,1}
end


function randomizeState(::Type{T}, sze::Int, rng::AbstractRNG ) where T
    Array{T}(rand(rng, (-1,1), sze)) 
end

function Ising(::Type{T}, sze::Int, rng::AbstractRNG=Random.default_rng()) where T
    Ising{T}(sze, zeros(T, sze), zeros(T, sze, sze), T(1), randomizeState(T, sze, rng))
end

function IsingCache_v0(::Type{T}, sze::Int) where T
    IsingCache_v0{T}(zeros(T, sze), zeros(T, sze), zeros(Bool,sze))
end

function randomFields!(ising::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    ising.H .= rand(rng, T, ising.sze).*2 .- 1
end

function randomWiring!(ising::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    ising.J .= randn(rng, T, ising.sze, ising.sze )
end 

function randomizeState!(ising::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    ising.s .= randomizeState(T, ising.sze, rng)
end

function parallelUpdate!(ising::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    h = ising.H + ising.J*ising.s
    if typeof(ising.H) <: CuArray
        r = CUDA.rand(T, ising.sze)
    else
        r = rand(rng, T, ising.sze)
    end
    ising.s .= 2*(2*ising.Beta*h .> log.(1 ./ r .- 1)) .- 1
    return nothing
end

function parallelUpdate_cpu!(ising::Ising{T}, ising_cache::IsingCache_v0{T}, rng::AbstractRNG=Random.default_rng()) where T
    h = ising_cache.h
    z = ising_cache.z
    r = ising_cache.r

    h .= ising.H
    mul!(h, ising.J, ising.s, T(1), T(1))
    rand!(rng, r)
    z .= 2*ising.Beta .* h .> log.(1 ./ r .- 1)
    ising.s .= T(2) .* z .- T(1)
    return nothing
end

function parallelUpdate_gpu!(ising::Ising{T}, ising_cache::IsingCache_v0{T}, rng::AbstractRNG=CURAND.default_rng()) where T
    h = ising_cache.h
    z = ising_cache.z
    r = ising_cache.r

    h .= ising.H
    mul!(h, ising.J, ising.s, T(1), T(1))
    rand!(rng, r)
    z .= 2*ising.Beta .* h .> log.(1 ./ r .- 1)
    ising.s .= T(2) .* z .- T(1)
    return nothing
end

function gpu(a::Array{T,N}) where {T,N}
    CuArray{T}(a)
end

function gpu(ising::Ising{T}) where{T}
    H = ising.H |> gpu
    J = ising.J |> gpu
    s = ising.s |> gpu
    Ising{T}(ising.sze, H, J, ising.Beta, s)
end

function gpu(ising_cache::IsingCache_v0{T}) where{T}
    ising_cache.
    z = ising_cache.z |> gpu
    h = ising_cache.h |> gpu
    r = ising_cache.r |> gpu
    IsingCache_v0{T}(h, r, z)
end

##########################
###  Mean Ising Model  ###
##########################

abstract type AbstractMeanIsingAlgorithm end


struct MeanIsingModel{T,Alg<:AbstractMeanIsingAlgorithm} 
    sze::Int
    m::AbstractArray{T,1}
    C::AbstractArray{T,2}
    D::AbstractArray{T,2}
    m_p::AbstractArray{T,1}
    C_p::AbstractArray{T,2}
    D_p::AbstractArray{T,2}
    algorithm::Alg
end

function MeanIsingModel(::Type{T}, sze::Int, alg::AbstractMeanIsingAlgorithm) where T
    MeanIsingModel{T, typeof(alg)}(
            sze,
            zeros(T, sze),
            zeros(T, sze, sze), 
            zeros(T, sze, sze), 
            zeros(T, sze),
            zeros(T, sze, sze), 
            zeros(T, sze, sze), 
            alg
            )
end

function initialize_state!(mean_ising::MeanIsingModel{T, Alg}, m::AbstractArray{T,1}) where {T, Alg}
    mean_ising.m .= m
    mean_ising.m_p .= copy(mean_ising.m)
    mean_ising.C .= diagm(1 .- m.^2)
    mean_ising.C_p .= copy(mean_ising.C)
    mean_ising.D .= zeros(T, mean_ising.sze, mean_ising.sze)
    mean_ising.D_p .= copy(mean_ising.D)
end

function initialize_state!(mean_ising::MeanIsingModel{T, Alg}) where {T, Alg}
    m .= zeros(T, mean_ising.sze)
    initialize_state!(mean_ising, m)
end


##########################
###  Plefka Expansions ###
##########################

abstract type AbstractIsingPlefkaExpansion{T} <: AbstractMeanIsingAlgorithm end

"""
Plefka[t-1,t] order 2
"""
struct IsingPlefka_t1_t{T} <: AbstractIsingPlefkaExpansion{T} 
    # cache::ComponentArray
end

function update_P!(ising::Ising{T}, mean_ising::MeanIsingModel{T,Alg}) where {T, Alg<:IsingPlefka_t1_t}
    mean_ising.m_p .= copy(mean_ising.m)
    mean_ising.m .=  update_m_P_t1_t_o2(ising.H, ising.J, mean_ising.m_p)
    mean_ising.C .= update_C_P_t1_t_o2(ising.H, ising.J, mean_ising.m, mean_ising.m_p)
    mean_ising.D .= update_D_P_t1_t_o2(ising.H, ising.J, mean_ising.m, mean_ising.m_p)
end


"""
Plefka[t] order 2
"""
struct IsingPlefka_t{T} <: AbstractIsingPlefkaExpansion{T} 
    # cache::ComponentArray
end

function update_P!(ising::Ising{T}, mean_ising::MeanIsingModel{T,Alg}) where {T, Alg<:IsingPlefka_t}
    mean_ising.m_p .= copy(mean_ising.m)
    mean_ising.C_p .= copy(mean_ising.C)
    mean_ising.m .= update_m_P_t_o2(ising.H, ising.J, mean_ising.m_p, mean_ising.C_p)
    mean_ising.C .= update_C_P_t_o2(ising.H, ising.J, mean_ising.m, mean_ising.C_p)
    mean_ising.D .= update_D_P_t_o2(ising.H, ising.J, mean_ising.m, mean_ising.m_p, mean_ising.C_p)
end

"""
Plefka[t-1] order 1
"""
struct IsingPlefka_t1{T} <: AbstractIsingPlefkaExpansion{T} 
    # cache::ComponentArray
end

function update_P!(ising::Ising{T}, mean_ising::MeanIsingModel{T,Alg}) where {T, Alg<:IsingPlefka_t1}
    mean_ising.m_p .= copy(mean_ising.m)
    mean_ising.C_p .= copy(mean_ising.C)
    mean_ising.m .= update_m_P_t1_o1(ising.H, ising.J, mean_ising.m_p)
    mean_ising.C .= update_C_P_t1_o1(ising.H, ising.J, mean_ising.m, mean_ising.m_p, mean_ising.C_p)
    mean_ising.D .= update_D_P_t1_o1(ising.H, ising.J, mean_ising.m_p, mean_ising.C_p)
end

"""
Plefka2[t] order 2
"""
struct IsingPlefka2_t{T} <: AbstractIsingPlefkaExpansion{T} 
    # cache::ComponentArray
end

function update_P!(ising::Ising{T}, mean_ising::MeanIsingModel{T,Alg}) where {T, Alg<:IsingPlefka2_t}
    mean_ising.m_p .= copy(mean_ising.m)
    mean_ising.C_p .= copy(mean_ising.C)
    mean_ising.D_p .= copy(mean_ising.D)
    m, D = update_D_P2_t_o2(ising.H, ising.J, mean_ising.m_p, mean_ising.C_p, mean_ising.D_p)
    mean_ising.m .= m
    mean_ising.D .= D
    mean_ising.C .= update_C_P2_t_o2(ising.H, ising.J, mean_ising.m, mean_ising.m_p, mean_ising.C_p)
end




