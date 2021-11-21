import UnrolledFlowNetworks as ufn
using ImageIO, Plots, FileIO
using Flux, NNlib
using OpticalFlowUtils

#root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift/"
root = "/home/nikopj/dataset/MPI_Sintel/training/clean/ambush_2"

I₀ = ufn.tensorload(joinpath(root,"frame_0001.png"), gray=true)
I₁ = ufn.tensorload(joinpath(root,"frame_0002.png"), gray=true)
@show size(I₀)

λ = 2e-2
J = 5
maxwarp = 5
maxit   = 1000
tol = 1e-4
tolwarp = 1e-5
pad = ufn.calcpad(size(I₀)[1:2], 2^J)
I₀ᵖ = pad_reflect(I₀, pad, dims=(1,2))
I₁ᵖ = pad_reflect(I₁, pad, dims=(1,2))

# v back-warps I1 to I0
v = ufn.flowctf(I₀ᵖ, I₁ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)
# ṽ back-warps I0 to I1
#ṽ = ufn.flowctf(I₁ᵖ, I₀ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)

# Wṽ = ufn.backward_warp(ṽ, v)
# error_v = ufn.pixelnorm(v - Wṽ)
# mask = error_v .< 7

WI₁ = ufn.backward_warp(I₁ᵖ, v) 
#WI₀ = ufn.backward_warp(I₀ᵖ, ṽ) 
x = cat(ufn.tensor2img.((I₀ᵖ, WI₁, I₁ᵖ))..., dims=3);

color_v = colorflow(permutedims(v[:,:,1,:], (3,1,2)));
# color_ṽ = colorflow(permutedims(ṽ[:,:,1,:], (3,1,2)));
# color_Wṽ = colorflow(permutedims(Wṽ[:,:,1,:], (3,1,2)));
# color = cat(color_v, color_Wṽ, color_ṽ, dims=3)

# maxflow = maximum(ufn.pixelnorm(v))
# CT = HSV{Float64}
# V = zeros(CT, size(v)[1:2]...)
# for i=1:size(V,1), j=1:size(V,2)
# 	V[i,j] = CT((180f0/π) * (π + atan(v[i,j,1,1], v[i,j,1,2])), norm(v[i,j,1,:]) / maxflow, 1)
# end

# maxflow = maximum(cat(ufn.pixelnorm.((v,Wṽ,ṽ))..., dims=3))
# color_v = colorflow(permutedims(v[:,:,1,:], (3,1,2)), maxflow);
# color_ṽ = colorflow(permutedims(ṽ[:,:,1,:], (3,1,2)), maxflow);
# color_Wṽ = colorflow(permutedims(Wṽ[:,:,1,:], (3,1,2)), maxflow);
# color = cat(color_v, color_Wṽ, color_ṽ, dims=3)


