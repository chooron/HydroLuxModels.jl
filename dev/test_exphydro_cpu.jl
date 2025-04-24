using CSV, DataFrames
using Lux
using Reactant, Enzyme
using Random
using BenchmarkTools

include("../src/HydroLuxModels.jl")

hidden_dims = 10
dev = cpu_device()
model = HydroLuxModels.ExpHydroLuxModel(hidden_dims)
ps, st = Lux.setup(Random.default_rng(), model) |> dev

tmp_input = rand(Float32, 3, hidden_dims, 1000) |> dev

output = model(tmp_input, ps, st)  # 0.005492 seconds (15 allocations: 544 bytes)
output[1] |> size