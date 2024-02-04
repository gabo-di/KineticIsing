using Random
using LinearAlgebra
using CUDA
using Infiltrator

function bool2int(x, val=0)
    v = 2^(length(x)-1)
    for i in eachindex(x)
        val += v*x[i]
        v >>= 1
    end
end

function bitfield(n, sze)
    digits(n, base=2, pad=sze) |> reverse
end

struct Ising_v1{T}
    sze::Int
    H::AbstractVector{T}
    J::AbstractMatrix{T}
    Beta::T
    s::AbstractVector{T}
end


Ising{T} = Ising_v1{T}

function randomizeState(::Type{T}, sze, rng::AbstractRNG ) where T
    Array{T}(rand(rng, (-1,1), sze)) 
end

function Ising(::Type{T}, sze, rng::AbstractRNG=Random.default_rng()) where T
    Ising{T}(sze, zeros(T, sze), zeros(T, sze, sze), T(1), randomizeState(T, sze, rng))
end

function randomFields!(self::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    self.H .= rand(rng, T, self.sze).*2 .- 1
end

function randomWiring!(self::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    self.J .= randn(rng, T, self.sze, self.sze )
end 

function randomizeState!(self::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    self.s = randomizeState(T, self.sze, rng)
end

function parallelUpdate!(self::Ising{T}, rng::AbstractRNG=Random.default_rng()) where T
    h = self.H + self.J*self.s
    if typeof(self.H) <: CuArray
        r = CUDA.rand(T, self.sze)
    else
        r = rand(rng, T, self.sze)
    end
    self.s .= 2*(2*self.Beta*h .> log.(1 ./ r .- 1)) .- 1
    return nothing
end

function parallelUpdate_cpu!(self::Ising{T}, h::Vector{T}, r::Vector{T}, z::Vector{Bool}, rng::AbstractRNG=Random.default_rng()) where T
    h .= self.H
    mul!(h, self.J, self.s, T(1), T(1))
    rand!(rng, r)
    z .= 2*self.Beta .* h .> log.(1 ./ r .- 1)
    self.s .= T(2) .* z .- T(1)
    return nothing
end

function parallelUpdate_gpu!(self::Ising{T}, h::CuArray{T,1}, r::CuArray{T,1}, z::CuArray{Bool,1}, rng::AbstractRNG=CURAND.default_rng()) where T
    h .= self.H
    mul!(h, self.J, self.s, T(1), T(1))
    rand!(rng, r)
    z .= 2*self.Beta .* h .> log.(1 ./ r .- 1)
    self.s .= T(2) .* z .- T(1)
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






