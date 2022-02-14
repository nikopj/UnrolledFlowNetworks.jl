module Utils

using Flux 
using NNlib 
using Statistics
import Zygote
import NNlib
using LazyGrids
import YAML
using FileIO
using CUDA

include("misc.jl")
include("conv.jl")

export loadargs, saveargs
export preprocess, postprocess, warp_bilinear
export pyramid, pyramid!, ConvGaussian, ConvGradient
export awgn

#=============================================================================
                     Preprocessing and Postprocessing
=============================================================================#

function calcpad(N::Int,s::Int)
	p = s*ceil(N/s) - N
	return ceil(Int, p/2), floor(Int, p/2)
end

function calcpad(N::NTuple{M,Int}, s::Int) where M
	Tuple([(calcpad.(N,s)...)...])
end

function unpad(x::AbstractArray, pad::NTuple{4,Int})
	@view x[begin+pad[1]:end-pad[2], begin+pad[3]:end-pad[4], :, :]
end

function preprocess(x::AbstractArray, y::AbstractArray, scales::Int)
	pad = calcpad(size(x)[1:2], 2^(scales-1))
	x = NNlib.pad_reflect(x, pad, dims=(1,2))
	y = NNlib.pad_reflect(y, pad, dims=(1,2))
	μ = mean(x+y, dims=(1,2))./2
	return x .- μ, y .- μ, (μ=μ, pad=pad)
end

function postprocess(v::AbstractArray, params::NamedTuple)
	return unpad(v, params.pad) 
end

#=============================================================================
                             Warping
=============================================================================#

function warp_nearest!(dst, img, flow)
	M, N, C, B = size(img)
	y, x, c, b = ndgrid(1:M, 1:N, 1:C, 1:B) .|> Zygote.dropgrad
	u = clamp.(round.(Int, y .+ selectdim(flow, 3, 1:1)), 1, M) 
	v = clamp.(round.(Int, x .+ selectdim(flow, 3, 2:2)), 1, N) 
	index = tuple.(u,v,c,b)
	NNlib.gather!(dst, img, index) 
end

function warp_nearest(img, flow) 
	dst = similar(img)
	warp_nearest!(dst, img, flow)
	return dst
end

function warp_bilinear(img::AbstractArray{T,4}, flow) where {T}
	if all(flow .== 0)
		return img
	end

	local index00, index01, index10, index11
	M, N, C, B = size(img)
	x, y, c, b = ndgrid(1:M, 1:N, 1:C, 1:B) .|> Zygote.dropgrad

	# points
	u = x .+ selectdim(flow, 3, 1:1) 
	v = y .+ selectdim(flow, 3, 2:2) 

	# nearby points
	u0 = clamp.(floor.(Int32, u), Int32(1), Int32(M)) 
	v0 = clamp.(floor.(Int32, v), Int32(1), Int32(N)) 
	u1 = clamp.(ceil.(Int32,  u), Int32(1), Int32(M)) 
	v1 = clamp.(ceil.(Int32,  v), Int32(1), Int32(N)) 

	Zygote.ignore() do
		index00 = CartesianIndex{4}.(u0, v0, c, b)
		index01 = CartesianIndex{4}.(u0, v1, c, b)
		index10 = CartesianIndex{4}.(u1, v0, c, b)
		index11 = CartesianIndex{4}.(u1, v1, c, b)
	end
	
	# function values
	f00 = NNlib.gather(img, index00) 
	f01 = NNlib.gather(img, index01) 
	f10 = NNlib.gather(img, index10) 
	f11 = NNlib.gather(img, index11) 
	
	# interpolation coefficients
	c00 = (u1 - u).*(v1 - v) 
	c01 = (u1 - u).*(v - v0) 
	c10 = (u - u0).*(v1 - v) 
	c11 = (u - u0).*(v - v0) 

	return c00.*f00 + c01.*f01 + c10.*f10 + c11.*f11
end

function warp_bilinear(img::AbstractArray{T,3}, flow) where {T}
	return warp_bilinear(img[:,:,:,:], flow)[:,:,:,1]
end

"""
    awgn(x::Union{AbstractArray,CuArray}, σ)
Additive Gaussian White Noise with noise-level σ.
"""
awgn(x::AbstractArray, σ::Union{AbstractArray,<:Real}) = x .+ σ.*randn(eltype(x), size(x)...), σ
awgn(x::Union{CuArray, SubArray{<:Any,<:Any,<:CuArray}}, σ::Union{AbstractArray,<:Real}) = x .+ σ.*CUDA.randn(eltype(x), size(x)...), σ

"""
    awgn(x, σ::Vector)
AWGN where noise-levels corresponding to each batch element are supplied.
"""
function awgn(x::Union{T, SubArray{<:Any,<:Any,T}}, σ::Vector) where {T<:Union{AbstractArray,CuArray}}
	@assert length(σ) == size(x, ndims(x)) "Noise-levels length must match batch-size"
	awgn(x, reshape(σ, ones(Int, ndims(x)-1)..., :))
end

"""
    awgn(x, (σ₁, σ₂))
AWGN over batched tensor x, where noise level is drawn independently and uniformly 
between (σ₁, σ₂) for each batch element of x.
"""
function awgn(x, I::NTuple{2,<:Number})
	@assert I[2] ≥ I[1] "I[2] ≱ I[1] ($(I[2]) ≱ $(I[1]))"
	σ = I[1] .+ (I[2]-I[1]).*rand(eltype(x), size(x,ndims(x)))
	awgn(x, σ)
end

end
