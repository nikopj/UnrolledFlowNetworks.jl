#=
utils.jl
=#

ST(x,τ) = sign.(x).*max.(0, abs.(x).-τ);       # soft-thresholding
pixeldot(x,y) = sum(x.*y, dims=(3,4))          # dot-product of 4D pixel vectors
pixelnorm(x) = sqrt.(sum(abs2, x, dims=(3,4))) # 2-norm on 4D image-tensor pixel-vectors
BT(x,τ) = max.(0, 1 .- τ./pixelnorm(x)).*x     # Block-thresholding of 4D pixel vectors

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

function preprocess(I₀::AbstractArray, I₁::AbstractArray, stride::Int)
	pad = calcpad(size(I₀)[1:2], stride)
	I₀ᵖ, I₁ᵖ = (I₀, I₁) .|> x->pad_reflect(x, pad, dims=(1,2))
	μ = mean(I₀ᵖ.+I₁ᵖ, dims=(1,2))./2
	return I₀ᵖ.-μ, I₁ᵖ.-μ, (μ, pad)
end

function postprocess(xₚ::AbstractArray, params)
	return unpad(xₚ .+ params[1], params[2])
end

#=============================================================================
                             Convolution
=============================================================================#

"""
    fdkernel(T::Type)

Return 2D forward difference convolution kernels (analysis and synthesi
s) for
use with NNlib's conv.
"""
function fdkernel(T::Type=Float32)
	W = zeros(T, 2,2,1,2)
	W[:,:,1,1] = [1  0;-1 0];
	W[:,:,1,2] = [1 -1; 0 0];
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

function gaussiankernel(T::Type, σ, m=ceil(Int,6σ-1))
	K = zeros(T, m, m, 1, 1)
	c = (m-1)/2
	for i=0:m-1, j=0:m-1
		K[i+1,j+1,1,1] = exp(-((i-c)^2 + (j-c)^2)/2σ^2)
	end
	return K./sum(K)
end
gaussiankernel(σ, m=ceil(Int,6σ-1)) = gaussiankernel(Float32, σ, m)

function ConvGaussian(σ; stride=1)
	h = gaussiankernel(σ)
	P = size(h,1)
	padl, padr = ceil(Int,(P-stride)/2), floor(Int,(P-stride)/2)
	pad = (padl, padr, padl, padr)
	H(x) = conv(x, repeat(h,1,1,1,size(x,3)); pad=pad, stride=stride, groups=size(x,3))
	return H
end

function sobelkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = 0.25*[1 2 1; 0 0 0; -1 -2 -1];
	W[:,:,1,2] = 0.25*[1 0 -1; 2 0 -2; 1 0 -1];
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

function ConvSobel(σ; stride=1)
	h = sobelkernel()
	H(x) = conv(x, repeat(h,1,1,1,size(x,3)÷2); pad=1, stride=stride, groups=size(x,3)÷2)
	return H
end

function cdkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = 0.5*[0 1 0; 0 0 0; 0 -1 0];
	W[:,:,1,2] = 0.5*[0 0 0; 1 0 -1; 0 0 0];
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

#=============================================================================
                             Warping
=============================================================================#

"""
backward image warping
Wimg[x] = img[x + v]
"""
function backward_warp(img::Union{T, SubArray{<:Any,<:Any,T}}, v::Union{T, SubArray{<:Any,<:Any,T}}) where {T<:Union{AbstractArray,CuArray}}
	Wimg = similar(img)
	itp  = OnCell()|>Natural|>Cubic|>BSpline
	for i=1:size(img,3), j=1:size(img,4)
		Iimg = extrapolate(interpolate(img[:,:,i,j], itp), 0)
		for m=1:size(Wimg,1), n=1:size(Wimg,2)
			vx = [m; n] .+ v[m,n,:,j]
			Wimg[m,n,i,j] = Iimg(vx...)
		end
	end
	return Wimg
end

#=============================================================================
                             Flow
=============================================================================#

function colorflow(flow::Array{T,4}, maxflow=maximum(mapslices(norm, flow, dims=3))) where {T}
    CT = HSV{Float32}
    color(x1, x2) = ismissing(x1) || ismissing(x2) ?
        CT(0, 1, 0) :
        CT(180f0/π * atan(x1, x2), norm((x1, x2)) / maxflow, 1)
    x1 = selectdim(flow, 3, 1)
    x2 = selectdim(flow, 3, 2)
    return color.(x1, x2)
end

