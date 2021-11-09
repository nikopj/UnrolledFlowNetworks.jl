#=
utils.jl
=#

ST(x,τ) = sign.(x).*max.(abs.(x).-τ, 0);       # soft-thresholding
pixeldot(x,y) = sum(x.*y, dims=(3,4))          # dot-product of 4D pixel vectors
pixelnorm(x) = sqrt.(sum(abs2, x, dims=(3,4))) # 2-norm on 4D image-tensor pixel-vectors

function pixelmatvec(A,x)
	M, N, C, B = size(A)
	y = zeros(eltype(A), M, N, C, 1)
	for m=1:size(A,1), n=1:size(A,2)
		y[m,n,:,1] = A[m,n,:,:] * x[m,n,:,1]
	end
	return y
end

function pixelmatmul(A,G)
	M, N, C₁, B = size(A)
	M, N, B, C₂ = size(G)
	D = zeros(eltype(A), M, N, C₁, C₂)
	for m=1:size(A,1), n=1:size(A,2)
		D[m,n,:,:] = A[m,n,:,:] * G[m,n,:,:]
	end
	return D
end

#=============================================================================
                            Image <-> Tensor
=============================================================================#

function tensor2img(A::Array{<:Real,2})
	tensor2img(Gray, A)
end

function tensor2img(A::Array{<:Real,4})
	if size(A)[3] == 1
		return tensor2img(A[:,:,1,1])
	end
	return tensor2img(RGB, permutedims(A[:,:,:,1], (3,1,2)))
end

function tensor2img(ctype, A::Array{<:Real}) 
	reinterpret(reshape, ctype{N0f8}, N0f8.(clamp.(A,0,1)))
end

function img2tensor(A)
	B = Float32.(reinterpret(reshape, N0f8, A) |> collect)
	if ndims(B) == 3
		B = permutedims(B, (2,3,1))
		B = reshape(B, size(B)..., 1)
	elseif ndims(B) == 2
		B = reshape(B, size(B)..., 1, 1)
	end
	return B
end

function tensorload(path; gray=false)
	img = load(path)
	if gray
		img = Gray.(img)
	end
	img2tensor(img)
end

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

function preprocess(y::AbstractArray, stride::Int)
	pad = calcpad(size(y)[1:2], stride)
	yₚ = pad_reflect(y, pad, dims=(1,2))
	μ = mean(yₚ, dims=(1,2))
	return yₚ .- μ, (μ, pad)
end

function postprocess(xₚ::AbstractArray, params)
	μ, pad = params
	return unpad(xₚ .+ μ, pad)
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

function sobelkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = 0.25*[1 2 1; 0 0 0; -1 -2 -1];
	W[:,:,1,2] = 0.25*[1 0 -1; 2 0 -2; 1 0 -1];
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
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
function backward_warp(img::Array{T,4}, v) where {T}
	Wimg = similar(img)
	itp  = OnCell()|>Natural|>Cubic|>BSpline
	Iimg = extrapolate(interpolate(img[:,:,1,1], itp), 0)
	for i=1:size(Wimg,1), j=1:size(Wimg,2)
		vx = [i; j] .+ v[i,j,1,:]
		Wimg[i,j,1,1] = Iimg(vx...)
	end
	return Wimg
end

