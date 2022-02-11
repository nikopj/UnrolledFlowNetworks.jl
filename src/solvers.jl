module Solvers

include("utils/Utils.jl")
using .Utils: warp_bilinear, preprocess, postprocess, cdkernel, pyramid, ConvGaussian

using CUDA
using Flux
using NNlib: upsample_bilinear
using LinearAlgebra
using Printf

export powermethod, flow_ictf, ST

function ST(x::T, τ) where {T}
	return sign(x)*min(zero(T), abs(x)-τ) # shrinkage-thresholding
end

function powermethod(A::Function, b::AbstractArray; maxit=100, tol=1e-3, verbose=true)
	r = zeros(maxit)
	λ, λ′= 0, 0
	flag = true         # error flag: tolerance not reached
	for k ∈ 1:maxit
		b = A(b)
		b = b ./ norm(b)
		λ = sum(b .* A(b))
		r[k] = abs(λ-λ′)
		if verbose
			@printf "k: %3d, |λ-λ′|= %.3e\n" k r[k]
		end
		if r[k] <= tol
			flag = false; 
			break
		end
		λ′ = λ
	end
	return λ, b, flag
end
powermethod(A::AbstractMatrix, b::AbstractVector; kws...) = powermethod(x->A*x, b; kws...)

function TVL1_BCA(u₁::U, u₂::U, λ::T, v̄::M=missing, w::M=missing; 
	              maxit=100, tol=1e-3, verbose=true) where {T, U<:AbstractArray{T,4}, M<:Union{Missing,U}}
	# cuda or cpu
	device = u₁ isa CuArray ? cu : identity

	# img shape
	m, n = size(u₁)[1:2]

	# step-sizes
	τ = 0.99 |> T
	σ = 0.99 |> T

	# init conv operators
	W  = cdkernel(T) 
	Wr = repeat(W,1,1,1,2)
	W  = W  |> device
	Wr = Wr |> device
	D  = Conv(Wr; pad=1, groups=2) 
	Dᵀ = ConvTranspose(Wr; pad=1, groups=2) 

	# init loop variables
	if ismissing(v̄)
		v̄ = zeros(T,m,n,2,1) |> device
	end
	if ismissing(w)
		w = zeros(T,m,n,4,1) |> device # dual var
	end
	v′ = v̄
	v  = v̄

	residual = zeros(maxit)  
	∇u₂ = conv(u₂, W; pad=1)               # (m,n,2,1)
	b  = u₂ - u₁ - sum(∇u₂.*v̄, dims=3)     # (m,n,1,1)
	α = sum(abs2, ∇u₂, dims=3)             # (m,n,1,1)
	a = ∇u₂ ./ (α .+ T(1e-7))
	η = τ.*α

	for k=1:maxit 
		# proximal gradient ascent on dual
		y = w + σ*D(2v - v′)
		w = y ./ max.(1, sqrt.(sum(abs2, y, dims=3))./λ) 

		v′ = v
		# proximal gradient descent on primal
		x = v - τ*Dᵀ(w)
		r = sum(∇u₂.*x, dims=3) + b
		v = x + a.*(ST.(r, η) - r)

		residual[k] = norm(v - v′)/norm(v′)
		if verbose
			@printf "%3d: r=%.3e \n" k residual[k] 
		end
		if k > 1 && residual[k] ≤ tol
			residual = residual[1:k]
			break
		end
	end
	return v, w, residual
end

function flow_ictf(u₁, u₂, λ, scales, v̄=missing, dualvar=missing; σ=1, warps=1, retflows=false, verbose=true, kws...) 
	local v
	u₁, u₂, preparams = preprocess(u₁, u₂, scales)

	if retflows
		flows = Matrix{typeof(u₁)}(undef, (scales, warps))
	end

	# construct Gaussian pyramid
	H = ConvGaussian(u₁, σ; stride=2)
	img_pyramid = (pyramid(u₁, scales, H), pyramid(u₂, scales, H))

	# coarse to fine
	for j ∈ scales:-1:1
		u₁, u₂ = img_pyramid[1][j], img_pyramid[2][j]

		# iterative warping
		for w ∈ 1:warps
			ū₂ = (j==scales && w==1) ? u₂ : warp_bilinear(u₂, v̄)
			v, dualvar, res = TVL1_BCA(u₁, ū₂, λ, v̄, dualvar; verbose=false, kws...)
			v̄ = v
			if verbose
				@printf "j=%02d | w=%02d | res[%03d]=%.2e \n" j w length(res) res[end] 
			end
			if retflows
				flows[j,w] = v
			end
		end

		# upscale flow and dual variables to finer scale
		if j > 1
			dualvar = 2*upsample_bilinear(dualvar, (2,2)) 
			v̄ = 2*upsample_bilinear(v̄, (2,2)) 
		end
	end
	if retflows
		return flows
	end
	return postprocess(v, preparams)
end

end
