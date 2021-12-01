using Flux
using NNlib: gather
using LazyGrids
import UnrolledFlowNetworks as ufn
using OpticalFlowUtils
using FileIO 
using CUDA
using Printf
using Statistics

device = gpu
root = "dataset/MPI_Sintel/training/clean/ambush_5/"

u₀ = ufn.tensorload(Float32, joinpath(root,"frame_0001.png"); gray=false) |> device
u₁ = ufn.tensorload(Float32, joinpath(root,"frame_0002.png"); gray=false) |> device
vgt  = ufn.tensorload(Float32, "dataset/MPI_Sintel/training/flow/ambush_5/frame_0001.flo") |> device
mask = ufn.tensorload(Float32, "dataset/MPI_Sintel/training/occlusions/ambush_5/frame_0001.png") |> device
mask = 1 .- mask
@show size(u₀)

# M, N, C, B = size(u₀)
# y, x, c, b = ndgrid(1:M, 1:N, 1:C, 1:B)
# 
# u = clamp.(round.(Int, y .+ selectdim(vgt, 3, 1)), 1, M) 
# v = clamp.(round.(Int, x .+ selectdim(vgt, 3, 2)), 1, N) 
# 
# index = tuple.(u,v,c,b)
# Wu = gather(u₁, index) 
#
Wu = ufn.warp_bilinear(u₁, vgt)
#out = Wu.*mask |> cpu |> ufn.tensor2img

@printf "Loss = %.3e" mean(abs, mask.*(Wu - u₀))
# P = plot(axis=nothing, layout=(1,4), size=(1600,400))
# imgs = ufn.tensor2img(cat(u₀, Wu.*mask, u₁, mask.*abs.(Wu-u₀), dims=4))
# 
# titlev = ["u₀", "Mask*Warp(u₁,vgt)", "u₁", "Mask*|Warp(u₁,vgt) - u₀|"]
# 
# for i ∈ 1:length(P)
# 	plot!(P[i], imgs[:,:,i])
# 	title!(P[i], titlev[i])
# end




