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

function TVL1_BCA(I₀, I₁, λ, v̄=missing; maxit=100, tol=1e-2, verbose=true)
	M, N = size(I₀)[1:2]

	W, Wᵀ = fdkernel(eltype(I₀))
	D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
	Dᵀ(z)= conv(pad_constant(z, (0,1,0,1), dims=(1,2)), Wᵀ);

	if ismissing(v̄)
		v̄ = zeros(M,N,1,2)
	end
	∇I₁ = D(I₁) |> x-> permutedims(x, (1,2,4,3))
	b   = I₁ - I₀ - pixeldot(∇I₁, v̄)
	
	τ = 0.99 / sqrt(8)
	σ = 0.99 / sqrt(8)

	α = pixeldot(∇I₁,∇I₁)
	a = ∇I₁./(α .+ 1e-7)
	η = τ*α

	v = copy(v̄)        # primal var
	w = zeros(M,N,2,2) # dual var
	r = zeros(maxit)   # primal residual
	
	k = 0
	while k == 0 || k < maxit && r[k] > tol
		# proximal gradient descent on primal
		x = v .- τ*Dᵀ(w)
		ρ = pixeldot(∇I₁, x) .+ b
		vᵏ = x .+ a.*(ST(ρ, η) .- ρ)   # v <- prox_τf(x)

		# proximal gradient ascent on dual
		y = w .+ σ*D(2vᵏ - v)
		w = y ./ max.(1, pixelnorm(y)/λ) # w <- prox_σg∗(y)

		k += 1
		r[k] = norm(vᵏ - v)/norm(vᵏ)
		v = vᵏ
		if verbose
			@printf "k: %3d | r= %.3e \n" k r[k] 
		end
	end
	return v, r[1:k]
end

function TVL1_VCA(I₀::Array{T,X}, I₁::Array{T,X}, λ, v̄=missing; maxit=100, tol=1e-2, verbose=true) where {T,X}
	M, N, C, _ = size(I₀)

	W, Wᵀ = fdkernel(eltype(I₀))
	D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
	Dᵀ(z)= conv(pad_constant(z, (0,1,0,1), dims=(1,2)), Wᵀ);

	if ismissing(v̄)
		v̄ = zeros(T,M,N,1,2)
	end
	v = copy(v̄)           # primal var
	w = zeros(T, M,N,2,2) # dual var
	z = zeros(T, M,N,1,C)
	u = zeros(T, M,N,1,C)
	r = zeros(T, maxit)   # primal residual

	λ = T(λ)
	τ = T(0.99 / sqrt(8))
	σ = T(0.99 / sqrt(8))
	ρ = T(2)

	# ADMM init
	# pixel-vector in dim 3, vertical/horizontal gradient in dim 4
	∇I₁ᵀ = D(permutedims(I₁, (1,2,4,3))) |> x-> permutedims(x, (1,2,4,3))
	@ein b[m,n,i,j] := ∇I₁ᵀ[m,n,i,k]*v̄[m,n,j,k]
	b = I₁ .- I₀ .- b |> x-> permutedims(x, (1,2,4,3))

	Avᵏ = zeros(T, M,N,1,C)
	t   = zeros(T, M,N,1,2)
	A = ∇I₁ᵀ
	Aᵀ = permutedims(∇I₁ᵀ, (1,2,4,3))
	eyemat = zeros(T, M,N,2,2)
	eyemat[:,:,1,1] .= 1; eyemat[:,:,2,2] .= 1

	@ein B[m,n,i,j] := Aᵀ[m,n,i,k]*A[m,n,k,j]
	B = eyemat .+ T(τ*ρ)*B 
	B⁻ = similar(B)
	d  = B[:,:,1,1].*B[:,:,2,2] .- B[:,:,1,2].*B[:,:,2,1]
	B⁻[:,:,1,1] = B[:,:,2,2]
	B⁻[:,:,2,2] = B[:,:,1,1]
	B⁻[:,:,1,2] = -B[:,:,1,2]
	B⁻[:,:,2,1] = -B[:,:,2,1]
	B⁻ = B⁻./d
	
	k = 0
	while k == 0 || k < maxit && r[k] > tol
		# proximal gradient descent on primal
		x = v .- τ*Dᵀ(w)
		@ein t[m,n,j,i]   := Aᵀ[m,n,i,k] * (b.-z.+u)[m,n,j,k]
		t = x .- τ*ρ*t
		@ein vᵏ[m,n,j,i]  := B⁻[m,n,i,k] * t[m,n,j,k]
		@ein Avᵏ[m,n,j,i] := A[m,n,i,k] * vᵏ[m,n,j,k]
		z = BT(Avᵏ .+ b .+ u, 1/ρ)
		u = u .+ Avᵏ .+ b .- z

		# proximal gradient ascent on dual
		y = w .+ σ*D(2vᵏ - v)
		w = y ./ max.(one(T), pixelnorm(y)./λ) # w <- prox_σg∗(y)

		k += 1
		r[k] = norm(vᵏ - v)/norm(vᵏ)
		v = vᵏ
		if verbose
			@printf "k: %03d | r= %.3e \n" k r[k] 
		end
	end
	return v, r[1:k]
end

function flowctf(I₀, I₁, λ, J; maxwarp=0, verbose=true, tol=1e-3, maxit=100, tolwarp=1e-4)
	C = size(I₀,3)
	if C == 1
		TVL1 = TVL1_BCA
	else
		TVL1 = TVL1_VCA
	end

	# construct gaussian pyramid 
	h = gaussiankernel(eltype(I₀),1)
	H(x) = permutedims(x, (1,2,4,3)) |> x->conv(x, h; stride=2, pad=(size(h,1)-1)÷2) |> x->permutedims(x, (1,2,4,3))
	pyramid = [(I₀,I₁)]
	for j ∈ 2:J
		push!(pyramid, H.(pyramid[j-1]))
	end

	# iterative warp and TVL1, coarse-to-fine
	v = zeros(eltype(I₀), size(pyramid[J][1])[1:2]..., 1, 2)
	for j ∈ J:-1:1
		i = 0; s = 0
		while i == 0 || i < maxwarp && s > tolwarp
			WI₁ = backward_warp(pyramid[j][2], v)
			vⁱ, r = TVL1(pyramid[j][1], WI₁, λ, v; maxit=maxit, tol=tol, verbose=false)
			s = norm(vⁱ - v)/norm(vⁱ)
			if verbose
				@printf "j= %02d | i= %02d | s= %.2e | k= %03d | r= %.2e \n" j-1 i s length(r) r[end]
			end
			i += 1
			v = vⁱ
		end
		if j > 1
			v = 2*upsample_bilinear(v, (2,2))
		end
	end
	return v
end
