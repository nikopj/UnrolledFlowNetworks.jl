#=
solvers.jl
=#

function powermethod(A::Function, b::AbstractArray; maxit=100, tol=1e-3, verbose=true)
	r = zeros(maxit)
	λ, λᵏ= 0, 0
	flag = true         # error flag: tolerance not reached
	for k ∈ 1:maxit
		b = A(b)
		b = b ./ mapslices(norm, b, dims=(1,2))
		λ = sum(b.*A(b))
		r[k] = abs(λ-λᵏ)
		λᵏ = λ
		if verbose
			@printf "k: %3d, |λ-λᵏ|= %.3e\n" k r[k]
		end
		if r[k] <= tol
			flag = false; 
			break
		end
	end
	return λ, b, flag
end

function TVL1_BCA(u₀::Array{T,4}, u₁::Array{T,4}, λ::Real, v̄=missing, w=missing; maxit=100, tol=1e-2, verbose=true) where {T}
	M, N, _, _ = size(u₀)

	λ = T(λ)
	# step-sizes
	τ = T(0.99 / sqrt(8))
	σ = T(0.99 / sqrt(8))

	# init conv operators
	W, _ = sobelkernel(T)
	Wd = repeat(W, 1,1,1,2)
	D  = Conv(Wd; pad=1, groups=2);
	Dᵀ = ConvTranspose(Wd; pad=1, groups=2);

	# init loop variables
	if ismissing(v̄)
		v̄ = zeros(T,M,N,2,1)
	end
	vᵏ = v̄
	v  = v̄
	if ismissing(w)
		w  = zeros(T,M,N,4,1) # dual var
	end
	residual = zeros(maxit)  

	∇u = conv(u₁, W; pad=1)               # (M,N,2,1)
	b  = u₁ - u₀ - sum(∇u.*v̄, dims=3)     # (M,N,1,1)
	α = sum(abs2, ∇u, dims=3) .+ T(1e-7)  # (M,N,1,1)
	η = τ.*α
	# v = ∇u.*(ST(b,η) - b)./α              # (M,N,2,1)

	k = 0
	while k == 0 || k < maxit && residual[k] > tol
		# proximal gradient ascent on dual
		y = w + σ*D(2v - vᵏ)
		w = y ./ max.(1, sqrt.(sum(abs2, y, dims=3))./λ) 
		#w = min.(λ, max.(-λ, y))

		vᵏ = v
		# proximal gradient descent on primal
		x = v - τ*Dᵀ(w)
		r = sum(∇u.*x, dims=3) + b
		v = x + ∇u.*(ST(r, η) - r)./α  

		k += 1
		residual[k] = norm(v - vᵏ)/norm(vᵏ)
		if verbose
			@printf "%3d: r=%.3e \n" k residual[k] 
		end
	end
	return v, w, residual[1:k]
end

function TVL1_VCA(u₀::Array{T,4}, u₁::Array{T,4}, λ, v̄=missing, dual_vars=missing; maxit=100, tol=1e-2, verbose=true) where {T}
	M, N, C, _ = size(u₀)

	λ = T(λ)
	# step-sizes
	τ = T(0.99 / sqrt(8))
	σ = T(0.99 / sqrt(8))
	ρ = T(2)
	η = τ/ρ

	# init conv operators
	W, _ = sobelkernel(T)
	Wd = repeat(W, 1,1,1,2)
	D  = Conv(Wd; pad=1, groups=2);
	Dᵀ = ConvTranspose(Wd; pad=1, groups=2);

	# init loop variables
	if ismissing(v̄)
		v̄ = zeros(T,M,N,2,1)
	end
	vᵏ = v̄
	v  = v̄
	if ismissing(dual_vars)
		t = zeros(T,M,N,C,1)
		s = zeros(T,M,N,C,1)
		w = zeros(T,M,N,4,1)
	else
		t, s, w = dual_vars
	end
	# t = zeros(T,M,N,C,1)
	# s = zeros(T,M,N,C,1)
	residual = zeros(maxit)  

	# ADMM init
	∇u = permutedims(u₁, (1,2,4,3)) |> x->conv(x, W; pad=1) # (M,N,2,C)
	A = permutedims(∇u, (1,2,4,3))

	# (M,N,2,2)
	# Q = AᵀA
	@ein Q[m,n,i,j] := A[m,n,k,i]*A[m,n,k,j]
	# R = 1/(I +τρAᵀA)
	Q .*= ρ
	Q[:,:,1,1] .+= 1
	Q[:,:,2,2] .+= 1
	R = similar(Q)
	R[:,:,1,1] =  Q[:,:,2,2]
	R[:,:,2,2] =  Q[:,:,1,1]
	R[:,:,1,2] = -Q[:,:,1,2]
	R[:,:,2,1] = -Q[:,:,2,1]
	d = Q[:,:,1,1].*Q[:,:,2,2] - Q[:,:,1,2].*Q[:,:,2,1]
	R = R./d

	# (M,N,C,1)
	@ein b[m,n,i,1] := A[m,n,i,k]*v̄[m,n,k,1]
	b = u₁ - u₀ - b 

	k = 0
	while k == 0 || k < maxit && residual[k] > tol
		# proximal gradient ascent on dual
		y = w + σ*D(2v - vᵏ)
		w = y ./ max.(1, sqrt.(sum(abs2, y, dims=3))./λ) 
		# w = min.(λ, max.(-λ, y))

		vᵏ = v
		# proximal gradient descent on primal
		x = v - τ*Dᵀ(w)
		# ADMM primal variable update
		# v = R(x - ρAᵀ(b - t + s))
		bts = b - t + s
		@ein temp[m,n,i,1] := A[m,n,k,i] * bts[m,n,k,1]
		temp = x - ρ*temp
		@ein v[m,n,i,1]  := R[m,n,i,k] * temp[m,n,k,1]

		@ein Av[m,n,i,1] := A[m,n,i,k] * v[m,n,k,1]
		t = BT(Av + b + s, η) # ADMM split variable update
		s = s + Av + b - t    # ADMM dual ascent

		k += 1
		residual[k] = norm(v - vᵏ)/norm(vᵏ)
		if verbose
			@printf "%3d: r=%.3e \n" k residual[k] 
		end
	end
	return v, (t,s,w), residual[1:k]
end

function flow_ictf(u₀::Array{T,4}, u₁::Array{T,4}, λ, J; maxwarp=0, verbose=true, tol=1e-3, maxit=100, tolwarp=1e-4) where {T}
	TVL1 = size(u₀,3)==1 ? TVL1_BCA : TVL1_VCA

	# construct Gaussian pyramid
	H = ConvGaussian(T, 1; stride=2)
	pyramid = [(u₀,u₁)]
	for j∈1:J
		push!(pyramid, H.(pyramid[j]))
	end

	v̄, dual_var = missing, missing
	v = nothing
	# coarse to fine
	for j ∈ J:-1:0
		w = 0; res = 0

		# iterative warping
		while w == 0 || w ≤ maxwarp && res > tolwarp
			u₀, u₁ = pyramid[j+1]
			ū₁ = j==J ? u₁ : warp_bilinear(u₁,v̄)
			v, dual_var, tvres = TVL1(u₀, ū₁, λ, v̄, dual_var; maxit=maxit, tol=tol, verbose=false)
			res = (w==0 && j==J) ? Inf : norm(v - v̄)/norm(v̄)
			if verbose
				@printf "j=%02d | w=%02d | res=%.2e | k=%03d | tvres[k]=%.2e \n" j w res length(tvres) tvres[end]
			end
			w += 1
			v̄ = v
		end

		# upscale flow and dual variables to finer scale
		if j>0 
			if typeof(dual_var) <: Tuple
				dual_var = dual_var .|> x->2*upsample_bilinear(x, (2,2))
			else
				dual_var = 2*upsample_bilinear(dual_var, (2,2)) 
			end
			v̄ = 2*upsample_bilinear(v̄, (2,2)) 
		end
	end
	return v̄
end

