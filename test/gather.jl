using Flux
using NNlib: gather
using LazyGrids
import UnrolledFlowNetworks as ufn
using OpticalFlowUtils
using FileIO 


root = "dataset/MPI_Sintel/training/clean/ambush_5/"

u₀ = ufn.tensorload(Float64, joinpath(root,"frame_0001.png"); gray=false)
u₁ = ufn.tensorload(Float64, joinpath(root,"frame_0002.png"); gray=false)
vgt  = load("dataset/MPI_Sintel/training/flow/ambush_5/frame_0001.flo") |> x->permutedims(convert(Array{Float32,3},x), (2,3,1)) |> Flux.unsqueeze(4)
mask = ufn.tensorload(Float64, "dataset/MPI_Sintel/training/occlusions/ambush_5/frame_0001.png")
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
Wu = ufn.warp_bilinear(u₁, vgt)
out = Wu.*mask |> ufn.tensor2img

P = plot(axis=nothing, layout=(1,4), size=(1600,400))
imgs = ufn.tensor2img(cat(u₀, Wu.*mask, u₁, mask.*abs.(Wu-u₀), dims=4))

titlev = ["u₀", "Mask*Warp(u₁,vgt)", "u₁", "Mask*|Warp(u₁,vgt) - u₀|"]

for i ∈ 1:length(P)
	plot!(P[i], imgs[:,:,i])
	title!(P[i], titlev[i])
end




