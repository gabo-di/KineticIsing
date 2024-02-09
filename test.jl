using Revise
includet("plefka.jl")
#####################################
#####################################
function f!(ising, n_it, rng, ising_cache)
    for i in 1:n_it
        # parallelUpdate!(ising, rng)
        if typeof(ising.H) <: CuArray
            parallelUpdate_gpu!(ising, ising_cache, rng)
        else
            parallelUpdate_cpu!(ising, ising_cache, rng)
        end
    end
end

using BenchmarkTools

rng = Random.default_rng();
Random.seed!(rng, 42);

s = 10000 
T = Float32
ising = Ising(T, s, rng);
ising_cache = IsingCache_v0(T, s);
randomFields!(ising, rng);
randomWiring!(ising, rng);

ising_gpu = ising |> gpu;
ising_cache_gpu = ising_cache |> gpu;
rng_gpu = CURAND.default_rng();
Random.seed!(rng_gpu, 42);


n = 1_000;
@benchmark f!($ising_gpu, $n, $rng_gpu, $ising_cache_gpu) # 11.56 s python is 22.8 s
@benchmark f!($ising, $n, $rng, $ising_cache) # 11.56 s python is 22.8 s

# using Profile
# using PProf
# Profile.clear()
# @profile f!(ising, n, rng, h, r, z)
# pprof()
#
#
# Profile.Allocs.clear()
# Profile.Allocs.@profile f!(ising_gpu, n, rng_gpu, h_gpu, r_gpu, z_gpu)
# PProf.Allocs.pprof()
