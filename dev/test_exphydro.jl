using CSV, DataFrames
using Lux
using Reactant, Enzyme
using Random
using BenchmarkTools
using ComponentArrays
using Optimisers
using Printf
using MLUtils: unsqueeze

include("../src/HydroLuxModels.jl")
Reactant.set_default_backend("cpu")

hidden_dims = 10
rng = Random.default_rng()
dev = reactant_device()
model = HydroLuxModels.ExpHydroLuxModel(hidden_dims)
ps, st = Lux.setup(rng, model) |> dev

# load data
file_path = "data/01013500.csv"
forcing_data = CSV.File(file_path) |> DataFrame
x_arr = forcing_data[!, [Symbol("prcp(mm/day)"), Symbol("tmean(C)"), Symbol("dayl(day)")]] |> Matrix |> permutedims
x_data = repeat(unsqueeze(x_arr, dims=2), 1, hidden_dims, 1) |> dev
y_data = forcing_data[!, [Symbol("flow(mm)")]] |> Matrix |> permutedims |> dev
tmp_input = rand(3, 10, 1000) |> dev
model_compiled = @compile sync = true model(tmp_input, ps, st)
# # All of this can be automated using the TrainState API
# train_state = Training.TrainState(model, ps, st, Adam(0.01f0))
# model_compiled = @compile sync = true model(x_data, ps, st)
# output = model_compiled(tmp_input, ps, st)  # 0.005492 seconds (15 allocations: 544 bytes)
# compiled_grad = @compile sync = true Enzyme.gradient(
#     Reverse, Const(sum ∘ first ∘ model), Const(tmp_input), ps, Const(st)
# )

# function train_model!(model, ps, st, x_data, y_data)
#     train_state = Lux.Training.TrainState(model, ps, st, Adam(0.01f0))
#     for iter in 1:1000
#         _, loss, _, train_state = Lux.Training.single_train_step!(
#             AutoEnzyme(), MSELoss(),
#             (x_data, y_data), train_state
#         )
#         if iter % 100 == 1 || iter == 1000
#             @printf "Iteration: %04d \t Loss: %10.9g\n" iter loss
#         end
#     end

#     return model, ps, st
# end

# train_model!(model, ps, st, x_data, y_data)