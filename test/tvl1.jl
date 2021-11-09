import UnrolledFlowNetworks as ufn
using ImageIO, Plots, FileIO
using Flux, NNlib
using OpticalFlowUtils

root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift"

I₀ = ufn.tensorload(joinpath(root,"00019.jpg"), gray=false)
I₁ = ufn.tensorload(joinpath(root,"00020.jpg"), gray=false)
@show size(I₀)

λ = 0.02
J = 5
maxwarp = 2
maxit   = 1000
tol = 1e-3
pad = ufn.calcpad(size(I₀)[1:2], 2^J)
I₀ᵖ = pad_reflect(I₀, pad, dims=(1,2))
I₁ᵖ = pad_reflect(I₁, pad, dims=(1,2))

# v back-warps I1 to I0
v = ufn.flowctf(I₀ᵖ, I₁ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)
# ṽ back-warps I0 to I1
ṽ = ufn.flowctf(I₁ᵖ, I₀ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)

Wṽ = ufn.backward_warp(ṽ, v)
error_v = ufn.pixelnorm(v - Wṽ)
mask = error_v .< 7

WI₁ = ufn.backward_warp(I₁ᵖ, v) 
WI₀ = ufn.backward_warp(I₀ᵖ, ṽ) 
x = cat(tensor2img.((I₀ᵖ, WI₁, WI₀, I₁ᵖ))..., dims=3);

maxflow = maximum(cat(ufn.pixelnorm.((v,Wṽ,ṽ))..., dims=3))
color_v = colorflow(permutedims(v[:,:,1,:], (3,1,2)), maxflow);
color_ṽ = colorflow(permutedims(ṽ[:,:,1,:], (3,1,2)), maxflow);
color_Wṽ = colorflow(permutedims(Wṽ[:,:,1,:], (3,1,2)), maxflow);
color = cat(color_v, color_Wṽ, color_ṽ, dims=3)


