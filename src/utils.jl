#=
utils.jl
=#

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

function preprocess(x::AbstractArray, y::AbstractArray, stride::Int)
	pad = calcpad(size(x)[1:2], stride)
	x = pad_reflect(x, pad, dims=(1,2))
	y = pad_reflect(y, pad, dims=(1,2))
	μ = mean(x+y, dims=(1,2))./2
	return x .- μ, y .- μ, (μ, pad)
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
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = [1  0	0;-1 0 0; 0 0 0];
	W[:,:,1,2] = [1 -1	0; 0 0 0; 0 0 0];
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

function ConvGaussian(T::Type, σ::Real=0.8; groups=1, stride=1, device=identity)
	P = ceil(Int, 6σ-1)
	if stride==2 && P%2!=0
		P += 1
	end
	h = gaussiankernel(T, σ, P) |> x->repeat(x, 1,1,1,groups) |> device
	P = size(h,1)
	pad = (P-stride)÷2
	padl, padr = ceil(Int,(P-stride)/2), floor(Int,(P-stride)/2)
	pad = (padl, padr, padl, padr)
	#H(x) = conv(x, h; pad=pad, stride=stride, groups=groups)
	return Conv(h, false; pad=pad, stride=stride, groups=groups)
end
ConvGaussian(σ::Real=0.8; kws...) = ConvGaussian(Float32, σ; kws...)

function sobelkernel(T::Type=Float32)
	W = zeros(T, 3,3,1,2)
	W[:,:,1,1] = 0.25*[1 2 1; 0 0 0; -1 -2 -1];
	W[:,:,1,2] = 0.25*[1 0 -1; 2 0 -2; 1 0 -1];
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

function ConvSobel(T::Type=Float32; stride=1)
	h = sobelkernel(T)[1]
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
function warp_interpolate(img::Union{T, SubArray{<:Any,<:Any,T}}, v::Union{T, SubArray{<:Any,<:Any,T}}) where {T<:Union{AbstractArray,CuArray}}
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

function warp_bilinear(img, flow)
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

#=============================================================================
                             MISC
=============================================================================#

# set value of key in nested dictionary
# key found greedily
# returns flag==true of key not found
function setrecursive!(d::Dict, key, value)
	if haskey(d, key)
		d[key] = value
		return false
	end
	flag = true
	for k ∈ keys(d)
		if d[k] isa Dict
			flag = setrecursive!(d[k], key, value)
			!flag && break
		end
	end
	return flag
end
