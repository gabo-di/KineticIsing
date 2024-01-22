using Revise
includet("kinetic_ising.jl")
#####################################
#####################################
function f!(ising, n_it, rng, h, r, z)
    for i in 1:n_it
        # parallelUpdate!(ising, rng)
        # parallelUpdate_v1!(ising, h, r, z, rng)
        if typeof(ising.H) <: CuArray
            parallelUpdate_gpu!(ising, h, r, z, rng)
        else
            parallelUpdate_cpu!(ising, h, r, z, rng)
        end
    end
end

using BenchmarkTools

rng = Random.default_rng();
Random.seed!(rng, 42);

s = 10000 
T = Float32
ising = Ising(T, s, rng);
randomFields!(ising, rng);
randomWiring!(ising, rng);
h = zeros(T, s);
r = zeros(T, s);
# z = BitArray(undef, s);
z = Vector{Bool}(undef, s);

ising_gpu = ising |> gpu;
h_gpu = gpu(h);
z_gpu = gpu(z);
r_gpu = gpu(r);
rng_gpu = CURAND.default_rng();
Random.seed!(rng_gpu, 42);


n = 1_000;
@benchmark f!($ising_gpu, $n, $rng_gpu, $h_gpu, $r_gpu, $z_gpu) # 11.56 s python is 22.8 s
@benchmark f!($ising, $n, $rng, $h, $r, $z) # 11.56 s python is 22.8 s

using Profile
using PProf
Profile.clear()
@profile f!(ising, n, rng, h, r, z)
pprof()


Profile.Allocs.clear()
Profile.Allocs.@profile f!(ising_gpu, n, rng_gpu, h_gpu, r_gpu, z_gpu)
PProf.Allocs.pprof()
