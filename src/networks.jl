#=============================================================================
                              Definitions 
=============================================================================#

struct BCANet{K,C,Cᵀ,T,S,R}
	A::NTuple{K,C}   # analysis conv
	Bᵀ::NTuple{K,Cᵀ} # synthesis conv
	τ::S             # log-domain primal step-size
	λ::NTuple{K,T}   # log-domain lagrange multiplier
	K::Int           # iterations
	M::Int           # number of filters 
	P::Int           # filter size (square side-length)
	s::Int           # conv stride
	∇::R             # spatial gradient operator
end
Flux.@functor BCANet
Flux.trainable(n::BCANet) = (n.A,n.Bᵀ,n.λ,n.τ)

struct PiBCANet{B,CI,CF}
	netarr::Vector{Vector{B}}
	blurimg::CI
	blurflo::CF
end
Flux.@functor PiBCANet
Flux.trainable(πn::PiBCANet) = (πn.netarr)

# aliases 
Base.getindex(πn::PiBCANet, i::Int, j::Int) = πn.netarr[i][j]
Base.getindex(πn::PiBCANet, i::Int) = πn.netarr[i]

function Base.getproperty(πn::PiBCANet, s::Symbol) 
	if s == :scales
		return length(πn.netarr)
	elseif s == :warps
		return length.(πn.netarr)
	elseif s == :shared_warp
		return all([all([πn[i,1] == πn[i,j] for j=1:πn.warps[i]]) for i=1:πn.scales])
	elseif s == :shared_scale
		return all(πn.warps .== πn.warps[1]) && 
		all([all([πn[1,j] == πn[i,j] for i=1:πn.scales]) for j=1:πn.warps[1]])
	elseif s ∈ [:K, :M, :P, :s]
		return getfield(πn[1,1], s)
	end
	return getfield(πn,s)
end

#=============================================================================
                              Constructors 
=============================================================================#

function BCANet(;K::Int=10, M::Int=16, P::Int=7, s::Int=1, λ₀=1f-1, W₀=missing, classical_init=false, lipschitz_init=true)
	# weight init
	if classical_init && ismissing(W₀)
		# Initialize with centeral difference kernels
		c = (P - 1) ÷ 2 + 1
		W₀ = zeros(Float32,P,P,2,M)
		W₀[c-1:c+1, c, 1, 1:2:M÷2] = repeat([-1; 0; 1], 1, 1, 1, M÷4)
		W₀[c, c-1:c+1, 1, 2:2:M÷2] = repeat([-1 0 1], 1, 1, 1, M÷4)
		W₀[c-1:c+1, c, 2, M÷2+2:2:end] = repeat([-1; 0; 1], 1, 1, 1, M÷4)
		W₀[c, c-1:c+1, 2, M÷2+1:2:end] = repeat([-1 0 1], 1, 1, 1, M÷4)
	elseif ismissing(W₀)
		W₀ = randn(Float32,P,P,2,M)
		W₀ .-= mean(W₀, dims=(1,2))
	end

	# conv op init
	padl, padr = ceil(Int,(P-s)/2), floor(Int,(P-s)/2)
	pad = (padl, padr, padl, padr)
	A = ntuple(i->Conv(copy(W₀), false; pad=pad, stride=s), K)
	Bᵀ= ntuple(i->ConvTranspose(copy(W₀), false; pad=pad, stride=s), K)

	if lipschitz_init
		L, _, flag = powermethod(x->Bᵀ[1](A[1](x)), randn(Float32,128,128,2,1), maxit=500, tol=1e-2, verbose=false)
		@show L, flag
		if L < 0
			@error "ERROR: BCANet: powermethod: L<0. Something is very very wrong..."
		end
		for k ∈ 1:K
			A[k].weight  ./= sqrt(L)
			Bᵀ[k].weight ./= sqrt(L)
		end
	end
	
	# threshold init 
	τ  = ntuple(i->log(0.95f0)*ones(Float32,1,1,1,1), K)
	λ  = ntuple(i->Float32(log(λ₀))*ones(Float32,1,1,M,1), K)

	# spatial gradient operator (non learned)
	∇ = ConvGradient()

	return BCANet(A, Bᵀ, τ, λ, K, M, P, s, ∇)
end

function PiBCANet(; scales::Int=1, warps::Union{Tuple,Vector,Int}=1, shared_warp::Bool=false, shared_scale::Bool=false, lipschitz_init=true, kws...)
	if warps isa Int
		warps = repeat([warps], scales)
	end
	@assert length(warps) == scales "warps indicates # warps-per-scales"

	# initialize network-array example network
	netarr  = Vector{Vector{BCANet}}(undef, scales)
	net1 = BCANet(; kws..., lipschitz_init=lipschitz_init)
	W₀ = net1.A[1].weight

	if shared_warp && shared_scale
		netvec = [net1 for w=1:warps[1]] 
		for j=1:scales
			netarr[j] = netvec
		end
	elseif shared_warp && !shared_scale
		for j=1:scales
			net1 = BCANet(; kws..., W₀=copy(W₀), lipschitz_init=false)
			netarr[j] = [net1 for w=1:warps[j]] 
		end
	elseif !shared_warp && shared_scale
		netvec = [BCANet(; kws..., W₀=copy(W₀), lipschitz_init=false) for w=1:warps[1]]
		for j=1:scales
			netarr[j] = netvec
		end
	elseif !shared_warp && !shared_scale
		for j=1:scales
			netarr[j] = [BCANet(; kws..., W₀=copy(W₀), lipschitz_init=false) for w=1:warps[j]]
		end
	end

	Himg = ConvGaussian(; groups=1, stride=2)
	Hflo = ConvGaussian(; groups=2, stride=2)
	return PiBCANet(netarr, Himg, Hflo)
end

#=============================================================================
                             Forward Method 
=============================================================================#
softST(x,τ) = x - τ*tanh(x/τ)                  # soft/differentiable ST
softclip(x,τ) = τ*tanh(x/τ)                    # soft/differentiable clipping

# Unrolled TVL1-BCA forward pass
function (net::BCANet)(u₁::T, u₂::T, v̄::T, w::T) where T <: AbstractArray
	∇u = net.∇(u₂)
	b  = u₂ - u₁ - sum(∇u.*v̄, dims=3)
	α  = sum(abs2, ∇u, dims=3) .+ 1f-7
	a  = ∇u ./ α
	v  = v̄
	for k ∈ 1:net.K
		# dual update
		w += net.A[k](v)
		w = softclip.(w, exp.(net.λ[k]))
		# primal update
		v -= net.Bᵀ[k](w)
		r = sum(∇u.*v, dims=3) + b
		v += a.*(softST.(r, exp.(net.τ[k]).*α) - r)
	end
	return v, w
end

function (πnet::PiBCANet)(u₁, u₂, v̄=missing; stopgrad=false, retflows=false)
	local v
	u₁, u₂, preparams = preprocess(u₁, u₂, πnet.scales)

	# construct flow buffer to store and return intermediate flows
	if retflows
		flows = [Zygote.Buffer(Vector{typeof(u₁)}(undef, πnet.warps[j])) for j in 1:πnet.scales]
	end

	# construct Gaussian pyramid
	img_pyramid = Zygote.ignore() do 
		(pyramid(u₁, πnet.scales, πnet.blurimg), pyramid(u₂, πnet.scales, πnet.blurimg))
	end

	# init variables
	# use similar to init with CuArrays of on CUDA
	m, n, _, B = size(img_pyramid[1][end])
	if ismissing(v̄)
		v̄ = similar(u₁, eltype(u₁), (m, n, 2, B))
		Zygote.ignore() do
			fill!(v̄,0)
		end
	end
	dualvar = similar(u₁, eltype(u₁), (m÷πnet.s, n÷πnet.s, πnet.M, B))
	Zygote.ignore() do
		fill!(dualvar,0)
	end

	# coarse to fine
	for j ∈ πnet.scales:-1:1
		u₁, u₂ = img_pyramid[1][j], img_pyramid[2][j]

		# iterative warping
		for w ∈ 1:πnet.warps[j]
			ū₂ = warp_bilinear(u₂, stopgrad ? Zygote.dropgrad(v̄) : v̄)
			v, dualvar = πnet[j,w](u₁, ū₂, v̄, dualvar)

			if retflows
				flows[j][w] = v
			end
			v̄ = v
		end

		# upscale flow and dual variables to finer scale
		if j > 1
			v̄ = 2*upsample_bilinear(v̄, (2,2))
			dualvar = 2*upsample_bilinear(dualvar, (2,2))
		end
	end

	if retflows
		return [copy(flows[j]) for j in 1:πnet.scales]
	end
	return postprocess(v, preparams)
end

function Train.weight_decay_penalty(net::PiBCANet)
	loss = zero(Float32)
	for j ∈ 1:net.scales
		for w ∈ 1:net.warps[j]
			loss += Train.weight_decay_penalty(net[j,w])
			net.shared_warp && break
		end
		net.shared_scale && break
	end
	return loss
end
function Train.weight_decay_penalty(net::BCANet)
	loss = zero(Float32)
	for k ∈ 1:net.K
		loss += sum(abs2, net.A[k].weight) + sum(abs2, net.Bᵀ[k].weight)
	end
	return loss
end


#=============================================================================
                                  Projection 
=============================================================================#

project!(t::AbstractArray) = clamp!(t, 0, Inf)
function project!(c::Union{Conv,ConvTranspose})
	c.weight ./= max.(1, sqrt.(sum(abs2, c.weight, dims=(1,2))))
	return nothing
end
function project!(n::BCANet)
	project!.((n.A..., n.Bᵀ...))
	return nothing
end
function Train.project!(πn::PiBCANet)
	for j in 1:πn.scales
		project!.(πn[j])
	end
	return nothing
end

#=============================================================================
                                  SHOW 
=============================================================================#

function Base.show(io::IO, n::BCANet)
	print(io, "BCANet(K=", n.K, 
	          ", M=",n.M, 
	          ", P=", n.P, 
	          ", s=", n.s, ")")
	nps = sum(length, params(n)); nps_unit = ""
	if nps > 1000
		nps = nps ÷ 1000
		nps_unit = "k"
	end
	printstyled(io, "  # ", nps, nps_unit*" parameters"; color=:light_black)
end

function Base.show(io::IO, πn::PiBCANet)
	print(io, "PiBCANet(scales=", πn.scales, 
	          ", warps=", πn.warps,
	          ", shared_scale=", πn.shared_scale, 
	          ", shared_warp=", πn.shared_warp, 
	          ", K=", πn.K, 
	          ", M=", πn.M, 
	          ", P=", πn.P, 
	          ", s=", πn.s, ")")
	# get parameter count
	nps = length(Flux.destructure(πn)[1])

	# get units
	nps_unit = ""
	if nps > 1e6
		nps = ceil(Int, nps/1e6)
		nps_unit = "M"
	elseif nps > 1e3
		nps = ceil(Int, nps/1e3)
		nps_unit = "k"
	end
	printstyled(io, "  # ", nps, nps_unit*" parameters"; color=:light_black)
end

