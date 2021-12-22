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

function TVL1_BCA(u₀, u₁, λ, v̄=missing, w=missing; γ=0, β=1, maxit=100, tol=1e-2, verbose=true, device=identity) 
	M, N, _, _ = size(u₀)
	T = eltype(u₀)

	λ = T(λ)
	# step-sizes
	τ = T(0.99)
	σ = T(0.99)

	# init conv operators
	W, _ = cdkernel(T) 
	Wd = repeat(W, 1,1,1,2) ./ sqrt(8f0)
	D  = Conv(Wd; pad=1, groups=2) |> device
	Dᵀ = ConvTranspose(Wd; pad=1, groups=2) |> device
	W = device(W)

	# init loop variables
	if ismissing(v̄)
		v̄ = zeros(T,M,N,2,1) |> device
	end
	vᵏ = v̄
	v  = v̄
	if ismissing(w)
		w  = zeros(T,M,N,4,1) |> device # dual var
	end
	residual = zeros(maxit)  

	∇u₁ = conv(u₁, W; pad=1)               # (M,N,2,1)
	∇u₀ = conv(u₀, W; pad=1)               # (M,N,2,1)
	b  = u₁ - u₀ - sum(∇u₁.*v̄, dims=3)     # (M,N,1,1)
	α = sum(abs2, ∇u₁, dims=3)             # (M,N,1,1)
	η = τ.*α

	# image driven regularization
	α₀ = sqrt.(sum(abs2, ∇u₀, dims=3))
	E = T.(1 .- sigmoid.(γ.*(α₀ .- β)))

	k = 0
	while k == 0 || k < maxit && residual[k] > tol
		# proximal gradient ascent on dual
		y = w + σ.*E.*D(2v - vᵏ)
		w = y ./ max.(1, sqrt.(sum(abs2, y, dims=3))./λ) 
		#w = min.(λ, max.(-λ, y))

		vᵏ = v
		# proximal gradient descent on primal
		x = v - τ*Dᵀ(E.*w)
		r = sum(∇u₁.*x, dims=3) + b
		#v = x + ∇u₁.*(ST(r, η) - r)./(α  .+ 1f-7)
		mask = abs.(r) .≤ η
		v = x - ∇u₁.*(mask.*r./(α .+ 1f-7) + (1 .- mask).*τ.*sign.(r))

		k += 1
		residual[k] = norm(v - vᵏ)/norm(vᵏ)
		if verbose
			@printf "%3d: r=%.3e \n" k residual[k] 
		end
	end
	return v, w, residual[1:k]
end

function pixel_admm_2matinv(A, ρ)
	@ein F[m,n,i,j,b] := A[m,n,k,i,b]*A[m,n,k,j,b] # F = AᵀA
	F .*= ρ                                        # F = ρAᵀA
	E = similar(A)
	CUDA.allowscalar() do                          # F = I + ρAᵀA
		F[:,:,1:2,1:2,:] .+= [1 0; 0 1]            # E = (I + ρAᵀA)^-1
		E[:,:,1,1,:] =  F[:,:,2,2,:]
		E[:,:,1,2,:] = -F[:,:,1,2,:]
		E[:,:,2,1,:] = -F[:,:,2,1,:]
		E[:,:,2,2,:] =  F[:,:,1,1,:]
		d = F[:,:,1,1,:].*F[:,:,2,2,:] - F[:,:,1,2,:].*F[:,:,2,1,:]
		E ./= d
	end
	return E
end

function TVL1_VCA(u₀, u₁, λ, v̄=missing, dual_vars=missing; γ=0, β=1, maxit=100, tol=1e-2, verbose=true, device=identity)
	M, N, C, _ = size(u₀)
	T = eltype(u₀)

	λ = T(λ)
	# step-sizes
	τ = T(0.99)
	σ = T(0.99)
	ρ = T(2)
	η = τ/ρ

	# init conv operators
	W, _ = cdkernel(T)
	Wd = repeat(W, 1,1,1,2) ./ T(sqrt(8))
	D  = Conv(Wd; pad=1, groups=2) |> device
	Dᵀ = ConvTranspose(Wd; pad=1, groups=2) |> device
	W = W |> device

	# init loop variables
	if ismissing(v̄)
		v̄ = zeros(T,M,N,2,1) |> device
	end
	vᵏ = v̄
	v  = v̄
	if ismissing(dual_vars)
		t = zeros(T,M,N,C,1) |> device
		s = zeros(T,M,N,C,1) |> device
		w = zeros(T,M,N,4,1) |> device
	else
		t, s, w = dual_vars
	end
	residual = zeros(maxit)  

	# image driven regularization
	∇u₀ = permutedims(u₀, (1,2,4,3)) |> x->conv(x, W; pad=1) # (M,N,2,C)
	α₀ = sqrt.(sum(abs2, ∇u₀, dims=(3,4)))
	# E = T.(exp.(-γ.*(α₀.^β)) .+ 1f-3)
	E = T.(1 .- sigmoid.(γ.*(α₀ .- β)))

	# ADMM init
	∇u = permutedims(u₁, (1,2,4,3)) |> x->conv(x, W; pad=1) # (M,N,2,C)
	A = permutedims(∇u, (1,2,4,3))

	# (M,N,2,2)
	# Q = AᵀA
	@ein Q[m,n,i,j] := A[m,n,k,i]*A[m,n,k,j]
	# R = 1/(I +τρAᵀA)
	Q .*= ρ
	R = similar(Q)
	CUDA.allowscalar() do
		Q[:,:,1,1] .+= 1
		Q[:,:,2,2] .+= 1
		R[:,:,1,1] =  Q[:,:,2,2]
		R[:,:,2,2] =  Q[:,:,1,1]
		R[:,:,1,2] = -Q[:,:,1,2]
		R[:,:,2,1] = -Q[:,:,2,1]
		d = Q[:,:,1,1].*Q[:,:,2,2] - Q[:,:,1,2].*Q[:,:,2,1]
		R = R./d
	end

	# (M,N,C,1)
	@ein b[m,n,i,1] := A[m,n,i,k]*v̄[m,n,k,1]
	b = u₁ - u₀ - b 

	k = 0
	while k == 0 || k < maxit && residual[k] > tol
		# proximal gradient ascent on dual
		y = w + σ.*E.*D(2v - vᵏ)
		w = y ./ max.(1f0, sqrt.(sum(abs2, y, dims=3))./λ) 
		# w = min.(λ, max.(-λ, y))

		vᵏ = v
		# proximal gradient descent on primal
		x = v - τ*Dᵀ(E.*w)
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

function flow_ictf(u₀, u₁, λ, J; retflows=false, maxwarp=0, verbose=true, tolwarp=1e-4, kws...) 
	TVL1 = size(u₀,3)==1 ? TVL1_BCA : TVL1_VCA

	device = (u₀ isa CuArray && CUDA.functional()) ? gpu : cpu

	# construct Gaussian pyramid
	H = ConvGaussian(eltype(u₀); groups=size(u₀,3), stride=2) |> device
	pyramid = [(u₀,u₁)]
	for j∈1:J
		push!(pyramid, H.(pyramid[j]))
	end

	if retflows
		flows = []
	end

	v̄, dual_var = missing, missing
	v = nothing
	# coarse to fine
	for j ∈ J:-1:0
		w = 0; res = 0

		# iterative warping
		while w == 0 || w ≤ maxwarp && res > tolwarp
			u₀, u₁ = pyramid[j+1]
			ū₁ = (j==J && w==0) ? u₁ : warp_bilinear(u₁,v̄)
			v, dual_var, tvres = TVL1(u₀, ū₁, λ, v̄, dual_var; verbose=false, device=device, kws...)
			res = (w==0 && j==J) ? Inf : norm(v - v̄)/norm(v̄)
			if verbose
				@printf "j=%02d | w=%02d | res=%.2e | k=%03d | tvres[k]=%.2e \n" j w res length(tvres) tvres[end]
			end
			w += 1
			v̄ = v
		end
		if retflows
			pushfirst!(flows, v)
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
	if retflows
		return flows
	end
	return v̄
end

