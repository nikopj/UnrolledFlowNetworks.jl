# using Plots
using ImageIO, FileIO, LinearAlgebra, Statistics, Printf
using CUDA
using UnrolledFlowNetworks, Flux
import UnrolledFlowNetworks as ufn
CUDA.allowscalar(false)

#root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift/"
root = "dataset/MPI_Sintel/training/clean/shaman_2/"

device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu
# device = cpu

gray = true
T = CUDA.functional() ? Float32 : Float64
# T = Float64

# ds_val = MPISintelDataset("dataset/MPI_Sintel/"; split="val", gray=gray)
# dl = Dataloader(ds_val, true; batch_size=1, crop_size=400, scale=0, J=J, σ=σ, device=device)
# F = dl[rand(1:length(dl))]
# @show F
# u₀ = F.frame0
# u₁ = F.frame1
# vgt = F.flows[1]
# 
# @show size(u₀), size(vgt)

u₀ = tensorload(T , joinpath(root,"frame_0048.png"); gray=gray) |> device
u₁ = tensorload(T , joinpath(root,"frame_0049.png"); gray=gray) |> device
vgt= tensorload(T , "dataset/MPI_Sintel/training/flow/shaman_2/frame_0048.flo") |> device
@show size(u₀), size(vgt)

λ = 2e-1
J = 6
kws = Dict(:retflows=>true, :γ=>100, :β=>Inf, :maxit=>10, :maxwarp=>5, :tol=>1e-3, :tolwarp=>1e-3)

#v, res = ufn.TVL1_VCA(u₀,u₁,λ; maxit=10, tol=1e-3, verbose=true)

u₀ᵖ, u₁ᵖ, params = ufn.preprocess(u₀, u₁, 2^J)
@show size(u₀ᵖ)
@show typeof(u₀ᵖ)

flows = flow_ictf(u₀ᵖ, u₁ᵖ, λ, J; kws...)
v = ufn.unpad(flows[1], params[2]) 

loss = ufn.EPELoss(v, vgt, device(ones(eltype(vgt), size(vgt))))
@printf "Loss = %.3f\n" loss
#end

# flows = colorflow(cat(v |> cpu |> collect,vgt |> cpu, dims=4), maximum(mapslices(norm, vgt|>cpu, dims=3)))

# P = plot(axis=nothing, layout=(2,1))
# titlev = ["TVL1", "gt"]
# 
# for i ∈ 1:length(P)
# 	plot!(P[i], flows[:,:,i])
# 	#title!(P[i], titlev[i])
# end


