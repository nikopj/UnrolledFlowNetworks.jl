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

	W, Wᵀ = cdkernel(eltype(I₀))
	D(v) = conv(v, W; pad=1);
	Dᵀ(w)= conv(w, Wᵀ; pad=1);

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

function TVL1_VCA(I₀, I₁, λ, v̄=missing; maxit=100, tol=1e-2, verbose=true)
	M, N, C, _ = size(I₀)

	W, Wᵀ = cdkernel(eltype(I₀))
	D(v) = conv(v, W; pad=1);
	Dᵀ(w)= conv(w, Wᵀ; pad=1);

	if ismissing(v̄)
		v̄ = zeros(M,N,1,2)
	end

	τ = 0.99 / sqrt(8)
	σ = 0.99 / sqrt(8)
	ρ = 2

	# pixel-vector in dim 3, vertical/horizontal gradient in dim 4
	∇I₁ᵀ = D(permutedims(I₁, (1,2,4,3))) |> x-> permutedims(x, (1,2,4,3))
	b    = I₁ .- I₀ .- pixelmatvec(∇I₁ᵀ, permutedims(v̄, (1,2,4,3)))

	A = ∇I₁ᵀ
	Aᵀ = permutedims(∇I₁ᵀ, (1,2,4,3))
	eyemat = zeros(M,N,2,2)
	eyemat[:,:,1,1] .= 1; eyemat[:,:,2,2] .= 1

	B  = eyemat .+ τ*ρ*pixelmatmul(Aᵀ,A)
	B⁻ = similar(B)
	d  = B[:,:,1,1].*B[:,:,2,2] .- B[:,:,1,2].*B[:,:,2,1]
	B⁻[:,:,1,1] = B[:,:,2,2]
	B⁻[:,:,2,2] = B[:,:,1,1]
	B⁻[:,:,1,2] = -B[:,:,1,2]
	B⁻[:,:,2,1] = -B[:,:,2,1]
	B⁻ = B⁻./(d .+ 1e-16)

	v = copy(v̄)        # primal var
	w = zeros(M,N,2,2) # dual var
	r = zeros(maxit)   # primal residual
	
	k = 0
	while k == 0 || k < maxit && r[k] > tol
		# proximal gradient descent on primal
		x = v .- τ*Dᵀ(w)
		vᵏ, s = pixelADMM(x, A, Aᵀ, B⁻, b, ρ, τ; maxit=2, tol=1e-3, verbose=false) # v<-prox_τf(x)

		# proximal gradient ascent on dual
		y = w .+ σ*D(2vᵏ - v)
		w = y ./ max.(1, pixelnorm(y)./λ) # w <- prox_σg∗(y)

		k += 1
		r[k] = norm(vᵏ - v)/norm(vᵏ)
		v = vᵏ
		if verbose
			@printf "k: %03d | r= %.3e | length(s)= %02d | s= %.2e \n" k r[k] length(s) s[end]
		end
	end
	return v, r[1:k]
end

function pixelADMM(v, A, Aᵀ, B⁻, b, ρ, τ; maxit=100, tol=1e-3, verbose=false) 
	M, N, C, _ = size(A)

	V = norm(v)
	r = zeros(maxit)
	v = permutedims(v, (1,2,4,3))
	x = zeros(M,N,2,1)
	z = zeros(M,N,C,1)
	u = zeros(M,N,C,1)

	k = 0
	while k == 0 || k < maxit && r[k] > tol
		x = pixelmatmul(B⁻, v .- τ*ρ*pixelmatmul(Aᵀ, b.- z.+u))
		Ax = pixelmatmul(A,x)
		z = BT(Ax .+ b .+ u, 1/ρ)
		t = Ax .+ b .- z
		u = u .+ t
		k += 1
		r[k] = maximum(pixelnorm(t))
		if verbose
			@printf "k: %3d | r= %.3e \n" k r[k] 
		end
	end
	return permutedims(x, (1,2,4,3)), r[1:k]
end

function flowctf(I₀, I₁, λ, J; maxwarp=0, verbose=true, tol=1e-3, maxit=100)
	C = size(I₀,3)
	if C == 1
		TVL1 = TVL1_BCA
	else
		TVL1 = TVL1_VCA
	end

	# construct gaussian pyramid 
	h = gaussiankernel(T,1)
	H(x) = permutedims(x, (1,2,4,3)) |> x->conv(x, h; stride=2, pad=(size(h,1)-1)÷2) |> x->permutedims(x, (1,2,4,3))
	pyramid = [(I₀,I₁)]
	for j ∈ 2:J
		push!(pyramid, H.(pyramid[j-1]))
	end

	# iterative warp and TVL1, coarse-to-fine
	v = zeros(size(pyramid[J][1])[1:2]..., 1, 2)
	for j ∈ J:-1:1
		i = 0; s = 0
		while i == 0 || i < maxwarp && s > tol
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
