using ImageIO, Plots, FileIO, LinearAlgebra, Statistics, Printf
using UnrolledFlowNetworks, Flux
import UnrolledFlowNetworks as ufn

#root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift/"
root = "dataset/MPI_Sintel/training/clean/shaman_2/"

#if true
gray = false
u₀ = tensorload(Float64, joinpath(root,"frame_0048.png"); gray=gray)
u₁ = tensorload(Float64, joinpath(root,"frame_0049.png"); gray=gray)
vgt= tensorload(Float64, "dataset/MPI_Sintel/training/flow/shaman_2/frame_0048.flo") 
@show size(u₀), size(vgt)

λ = 1e-1
J = 5
kws = Dict(:maxit=>1000, :maxwarp=>5, :tol=>1e-3, :tolwarp=>1e-3)

#v, res = ufn.TVL1_VCA(u₀,u₁,λ; maxit=10, tol=1e-3, verbose=true)

u₀ᵖ, u₁ᵖ, params = ufn.preprocess(u₀, u₁, 2^J)
@show size(u₀ᵖ)

v = flow_ictf(u₀ᵖ, u₁ᵖ, λ, J; kws...)
v = ufn.unpad(v, params[2]) |> collect

loss = ufn.EPELoss(v, vgt, ones(eltype(vgt), size(vgt)))
@printf "Loss = %.3f\n" loss
#end

flows = colorflow(cat(v,vgt, dims=4), maximum(mapslices(norm, vgt, dims=3)))

P = plot(axis=nothing, layout=(2,1))
titlev = ["TVL1", "gt"]

for i ∈ 1:length(P)
	plot!(P[i], flows[:,:,i])
	#title!(P[i], titlev[i])
end


