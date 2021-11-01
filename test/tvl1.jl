import UnrolledFlowNetworks as ufn
using ImageIO, Plots, FileIO
using Flux, NNlib
using OpticalFlowUtils

root = "/home/nikopj/dataset/DAVIS/JPEGImages/480p/chairlift"

I₀ = ufn.tensorload(joinpath(root,"00019.jpg"), gray=true)
I₁ = ufn.tensorload(joinpath(root,"00020.jpg"), gray=true)
@show size(I₀)

J = 5
pad = ufn.calcpad(size(I₀)[1:2], 2^J)
I₀ᵖ = pad_reflect(I₀, pad, dims=(1,2))
I₁ᵖ = pad_reflect(I₁, pad, dims=(1,2))
@show size(I₀ᵖ)
v, H = ufn.optical_flow(I₀ᵖ, I₁ᵖ, 20, J)

#K = ufn.gaussiankernel(eltype(I₀), 1)
#p = (size(K,1)-1) ÷ 2
#
#for i=1:3
#	global I₀, I₁
#	I₀ = conv(I₀, K; stride=2, pad=p)
#	I₁ = conv(I₁, K; stride=2, pad=p)
#end
#@show size(I₀)
#
#W, _ = ufn.fdkernel(eltype(I₀))
#D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
#
#∇I = D(I₀)
#Iₜ = I₁ - I₀
#
#v, r = ufn.TVL1(∇I, Iₜ, 20; maxit=1000, tol=1e-4)

WI₁ = ufn.backward_warp(I₁ᵖ[:,:,1,1], v) |> x->reshape(x, size(x)...,1,1)
x = cat(I₀ᵖ, WI₁, I₁ᵖ, dims=3);

# vc = v[:,:,1,1] + im*v[:,:,2,1]
# println(maximum(abs.(vc)))
# println(minimum(abs.(vc)))

#plot(colorflow(permutedims(v[:,:,:,1], (3,1,2))))

