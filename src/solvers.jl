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

function TVL1(a, b, λ, init=missing; maxit=100, tol=1e-2, verbose=true)
	M, N, P, _ = size(a)
	a = permutedims(a, (1,2,4,3)) # move channels into batch dimension
	
	#objfun = (x, Dx) -> norm(pixelnorm(Dx),1) + λ*norm(pixeldot(a,x).+b,1)
	τ = 0.99 / sqrt(8)
	σ = 0.99 / sqrt(8)

	α = pixeldot(a,a)
	ā = a./(α .+ 1e-7)
	η = τ*λ*α

	W, Wᵀ = fdkernel(eltype(a))
	D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
	Dᵀ(y)= conv(pad_constant(y, (0,1,0,1), dims=(1,2)), Wᵀ);

	if !ismissing(init)
		x, y = init
	else
		x = zeros(M,N,1,2) # primal var
		y = zeros(M,N,2,2) # dual var
	end
	r = zeros(maxit)   # primal residual
	#obj = zeros(maxit)   # primal residual
	
	k = 0
	while k == 0 || k < maxit && r[k] > tol

		# proximal gradient descent on primal
		z = x .- τ*Dᵀ(y)
		ρ = pixeldot(a, z) .+ b
		xᵏ = z .+ ā.*(ST(ρ, η) .- ρ)   # x <- prox_τF(z)

		# proximal gradient ascent on dual
		w = y .+ σ*D(2xᵏ - x)
		#y = min.(1, max.(-1, w))
		y = w ./ max.(1, pixelnorm(w)) # y <- prox_σG*(w)

		k += 1
		r[k] = norm(xᵏ - x)/norm(xᵏ)
		x = xᵏ
		#obj[k] = objfun(x, D(x))
		if verbose
			@printf "k: %3d | r= %.3e | obj= %.3e \n" k r[k] obj[k]
		end
	end
	return permutedims(x, (1,2,4,3)), y, r[1:k]
end

function optical_flow(I₀::Array{T,N}, I₁::Array{T,N}, λ, J) where {T,N}
	# compute gaussian pyramid
	# iterative: solve, upscale, warp, repeat
	K = gaussiankernel(T,1)
	S, _ = sobelkernel(T)
	B(x) = conv(x, K; stride=2, pad=(size(K,1)-1)÷2)
	D(x) = conv(x, S; pad=1)
	H = []
	V = []
	
	BI = [(I₀,I₁)]
	for j ∈ 1:J
		push!(BI, B.(BI[j]))
	end
	for j ∈ J+1:-1:1
		if j==J+1
			Upv = 0
			WI = BI[j][2]
			init = missing
		else
			global v, ṽ, w
			Upv = 2*upsample_bilinear(v, (2,2))
			ṽ = upsample_bilinear(permutedims(ṽ,(1,2,4,3)), (2,2))
			w = upsample_bilinear(w, (2,2))
			WI = backward_warp(BI[j][2][:,:,1,1], Upv)
			WI = reshape(WI, size(WI)..., 1,1)
			init = (ṽ,w)
		end
		∇I = D(BI[j][1])
		Iₜ = WI .- BI[j][1]
		ṽ, w, r = TVL1(∇I, Iₜ, λ, init; maxit=500, tol=1e-3, verbose=false)
		v = ṽ .+ Upv
		h = (r[end], length(r))
		@show j h
		push!(H, h)
	end
	return v, H
end
