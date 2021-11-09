import UnrolledFlowNetworks as ufn
using ImageIO, Plots, FileIO
using Flux, NNlib
using OpticalFlowUtils

root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift"

I₀ = ufn.tensorload(joinpath(root,"00019.jpg"), gray=true)
I₁ = ufn.tensorload(joinpath(root,"00020.jpg"), gray=true)
@show size(I₀)

λ = 30
J = 5
maxwarp = 2
maxit   = 1500
tol = 1e-4
pad = ufn.calcpad(size(I₀)[1:2], 2^J)
I₀ᵖ = pad_reflect(I₀, pad, dims=(1,2))
I₁ᵖ = pad_reflect(I₁, pad, dims=(1,2))

# v back-warps I1 to I0
v = ufn.flowctf(I₀ᵖ, I₁ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)
# ṽ back-warps I0 to I1
ṽ = ufn.flowctf(I₁ᵖ, I₀ᵖ, λ, J; maxwarp=maxwarp, maxit=maxit, tol=tol)

Wṽ₁ = ufn.backward_warp(ṽ[:,:,:,1:1], v)
Wṽ₂ = ufn.backward_warp(ṽ[:,:,:,2:2], v) 
Wṽ  = cat(Wṽ₁, Wṽ₂, dims=4)

error_v = ufn.pixelnorm(v - Wṽ)
mask = error_v .< 5

WI₁ = ufn.backward_warp(I₁ᵖ, v) 
WI₀ = ufn.backward_warp(I₀ᵖ, ṽ) 
x = cat(I₀ᵖ, WI₁, I₁ᵖ, dims=3);
y = cat(I₀ᵖ, WI₀, I₁ᵖ, dims=3);

color_v = colorflow(permutedims(v[:,:,1,:], (3,1,2)));
color_ṽ = colorflow(permutedims(ṽ[:,:,1,:], (3,1,2)));
color_Wṽ = colorflow(permutedims(Wṽ[:,:,1,:], (3,1,2)));
color = cat(color_v, color_Wṽ, color_ṽ, dims=3)


