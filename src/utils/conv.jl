#=============================================================================
                             Kernels
=============================================================================#

transpose_kernel(W::AbstractArray{T,4}) where {T} = reverse(permutedims(W, (2,1,4,3)), dims=:) 

# forward difference kernel
function fdkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = [1  0 0;-1 0 0; 0 0 0];
	W[:,:,1,2] = [1 -1 0; 0 0 0; 0 0 0];
	return W
end

# centered difference kernel
function cdkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = 0.5*[0 1 0; 0 0 0; 0 -1 0];
	W[:,:,1,2] = 0.5*[0 0 0; 1 0 -1; 0 0 0];
	return W
end

function gaussiankernel(T::Type, σ::Real, m::Int=ceil(Int,6σ-1))
	K = zeros(T, m, m, 1, 1)
	c = (m-1)/2
	for i=0:m-1, j=0:m-1
		K[i+1,j+1,1,1] = exp(-((i-c)^2 + (j-c)^2) / (2σ^2))
	end
	return K./sum(K)
end
gaussiankernel(σ, m=ceil(Int,6σ-1)) = gaussiankernel(Float32, σ, m)

function ConvGaussian(T::Type, σ::Real=1; groups=1, stride=1)
	k = ceil(Int, 6σ-1)
	if stride == 2 && k%2 != 0
		k += 1
	end
	K = gaussiankernel(T, σ, k) |> x -> repeat(x, 1, 1, 1, groups)
	k = size(K,1)
	padl, padr = ceil(Int,(k-stride)/2), floor(Int,(k-stride)/2)
	pad = (padl, padr, padl, padr)
	return Conv(K, false; pad=pad, stride=stride, groups=groups)
end
ConvGaussian(σ::Real=1; kws...) = ConvGaussian(Float32, σ; kws...)

function  ConvGaussian(x::AbstractArray{T,4}, σ::Real=1; stride=1) where {T}
	H = ConvGaussian(T, σ; groups=size(x,3), stride=stride)
	if x isa CuArray
		H = H |> Flux.gpu
	end
	return H
end

function ConvGradient(T::Type=Float32; groups=1, stride=1)
	K = cdkernel(T) |> x -> repeat(x, 1, 1, 1, groups)
	k = size(K,1)
	padl, padr = ceil(Int,(k-stride)/2), floor(Int,(k-stride)/2)
	pad = (padl, padr, padl, padr)
	return Conv(K, false; pad=pad, stride=stride, groups=groups)
end

function  ConvGradient(x::AbstractArray{T,4}; stride=1) where {T}
	G = ConvGaussian(T, σ; groups=size(x,3), stride=stride)
	if x isa CuArray
		G = G |> Flux.gpu
	end
	return G
end

#=============================================================================
                             Pyramid
=============================================================================#

function pyramid!(buf::AbstractVector, x::AbstractArray, convop)
	buf[1] = x
	for j ∈ 2:length(buf)
		buf[j] = convop(buf[j-1])
	end
	return buf
end

function pyramid(x::AbstractArray, scales::Int, convop)
	buf = Vector{typeof(x)}(undef, scales)
	return pyramid!(buf, x, convop)
end

function pyramid(x::AbstractArray{T,N}, scales::Int, σ::Real=1) where {T,N}
	H = ConvGaussian(T, σ; groups=size(x,3), stride=2)
	return pyramid(x, scales, H)
end

