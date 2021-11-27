using ImageIO, Plots, FileIO, LinearAlgebra, Statistics
using UnrolledFlowNetworks, Flux
import UnrolledFlowNetworks as ufn

#root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift/"
root = "dataset/MPI_Sintel/training/clean/shaman_2/"

u₀ = ufn.tensorload(Float64, joinpath(root,"frame_0048.png"); gray=false)
u₁ = ufn.tensorload(Float64, joinpath(root,"frame_0049.png"); gray=false)
vgt  = load("dataset/MPI_Sintel/training/flow/shaman_2/frame_0048.flo") |> x->permutedims(convert(Array{Float32,3},x), (2,3,1)) |> Flux.unsqueeze(4)
@show size(u₀)
#@show size(v), typeof(v)

λ = 1e-1
J = 5
kws = Dict(:maxit=>500, :maxwarp=>5, :tol=>1e-2, :tolwarp=>1e-3)

#v, res = ufn.TVL1_VCA(u₀,u₁,λ; maxit=10, tol=1e-3, verbose=true)

u₀ᵖ, u₁ᵖ, params = ufn.preprocess(u₀, u₁, 2^J)
@show size(u₀ᵖ)

v = flow_ictf(u₀ᵖ, u₁ᵖ, λ, J; kws...)
v = ufn.unpad(v, params[2]) |> collect

# loss = ufn.EPELoss(v̂, v, ones(Float32, size(v)))
# @show loss


