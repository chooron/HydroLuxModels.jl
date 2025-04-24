using CSV, DataFrames
using Lux
using Reactant, Enzyme
using Random
using BenchmarkTools

include("../src/HydroLuxModels.jl")

hidden_dims = 10
dev = reactant_device()
model = HydroLuxModels.ExpHydroLuxModel(hidden_dims)
ps, st = Lux.setup(Random.default_rng(), model) |> dev

tmp_input = rand(Float32, 3, hidden_dims, 1000) |> dev

model_compiled = @compile sync = true model(tmp_input, ps, st)

@btime output = model_compiled(tmp_input, ps, st)  # 0.005492 seconds (15 allocations: 544 bytes)
compiled_grad = @compile sync = true Enzyme.gradient(
    Reverse, Const(sum ∘ first ∘ model), Const(tmp_input), ps, Const(st)
)

compiled_grad(
    Reverse, Const(sum ∘ first ∘ model), Const(tmp_input), ps, Const(st)
) # 0.019394 seconds (47 allocations: 1.328 KiB)

# All of this can be automated using the TrainState API
train_state = Training.TrainState(model, ps, st, Adam(0.01f0))

gs, loss, stats, train_state = Training.single_train_step!(
    AutoEnzyme(), MSELoss(),
    (x, dev(rand(rng, Float32, 10, 2))), train_state
)