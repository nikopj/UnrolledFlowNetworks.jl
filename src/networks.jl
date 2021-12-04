#=
networks.jl
=#

#=============================================================================
                              Definitions 
=============================================================================#

struct BCANet{K,C,Cᵀ,T,S,R}
	A::NTuple{K,C}   # analysis conv
	Bᵀ::NTuple{K,Cᵀ} # synthesis conv
	τ::S             # primal step-size
	λ::NTuple{K,T}   # lagrange multiplier
	K::Int           # iterations
	M::Int           # number of filters / 2
	P::Int           # filter size (square side-length)
	s::Int           # conv stride
	∇::R             # spatial gradient operator
end
Flux.@functor BCANet
Flux.trainable(n::BCANet) = (n.A,n.Bᵀ,n.λ,n.τ)

struct PiBCANet{K,N}
	net::NTuple{K,N}
end
Flux.@functor PiBCANet
Flux.trainable(πn::PiBCANet) = (πn.net)

# aliases 
function Base.getproperty(πn::PiBCANet, s::Symbol) 
	if s == :W
		return length(πn.net) - 1
	elseif s == :shared
		return πn.net[1] == πn.net[end]
	elseif s ∈ [:K, :M, :P, :s]
		return getfield(πn.net[1], s)
	end
	return getfield(πn,s)
end
Base.getindex(πn::PiBCANet, k) = πn.net[k]

#=============================================================================
                              Constructors 
=============================================================================#

function BCANet(;K::Int=10, M::Int=16, P::Int=7, s::Int=1, λ₀=1f-1, W₀=randn(Float32,P,P,2,M), init=true)
	padl, padr = ceil(Int,(P-s)/2), floor(Int,(P-s)/2)
	pad = (padl, padr, padl, padr)
	A = ntuple(i->Conv(copy(W₀), false; pad=pad, stride=s), K)
	Bᵀ= ntuple(i->ConvTranspose(copy(W₀), false; pad=pad, stride=s), K)
	if init
		L, _, flag = powermethod(x->Bᵀ[1](A[1](x)), randn(Float32,128,128,2,1), maxit=500, tol=1e-2, verbose=false)
		if L < 0
			println("ERROR: BCANet: powermethod: L<0. Something is very very wrong...")
		end
		@show L, flag
		for k ∈ 1:K
			A[k].weight  ./= sqrt(L)
			Bᵀ[k].weight ./= sqrt(L)
		end
	end
	τ = ntuple(i->1f0*ones(Float32,1,1,1,1), K)
	λ = ntuple(i->Float32(λ₀)*ones(Float32,1,1,M,1), K)
	∇ = Conv(sobelkernel()[1], false; pad=1)
	return BCANet(A, Bᵀ, τ, λ, K, M, P, s, ∇)
end

function PiBCANet(; W::Int=0, shared::Bool=true, init=true, kws...)
	net₀ = BCANet(; kws..., init=init)
	if shared
		net = ntuple(w->net₀, W+1)
	else
		W₀ = copy(net₀.A[1].weight)
		net = ntuple(w->BCANet(; kws..., W₀=W₀, init=false), W+1)
	end
	return PiBCANet(net)
end

#=============================================================================
                             Forward Method 
=============================================================================#

# Unrolled TVL1-BCA 
function (net::BCANet)(u₀, u₁, v̄, w)
	∇u = net.∇(u₁)
	b  = u₁ - u₀ - sum(∇u.*v̄, dims=3)
	α  = sum(abs2, ∇u, dims=3) .+ 1f-7
	∇ū = ∇u./α
	vᵏ = v̄
	v  = v̄
	for k ∈ 1:net.K
		# dual update
		w = min.(net.λ[k], max.(-net.λ[k], w + net.A[k](2v - vᵏ)))
		# primal update
		vᵏ= v
		x = v - net.τ[k].*net.Bᵀ[k](w)
		r = sum(abs2, ∇u.*x, dims=3) + b
		v = x + ∇ū.*(ST(r, net.τ[k].*α) - r)
	end
	return v, w
end

function (πnet::PiBCANet)(u₀, u₁, J::Int; stopgrad=false, retflows=false, blur_op=missing)
	u₀, u₁, preparams = preprocess(u₀, u₁, 2^(J+πnet.s÷2))

	# construct Gaussian pyramid
	pyramid = Matrix{typeof(u₀)}(undef, (J+1,2))
	Zygote.ignore() do 
		if ismissing(blur_op)
			blur_op = ConvGaussian(1; groups=size(u₀,3), stride=2, device=(u₀ isa CuArray) ? gpu : cpu)
		end
		pyramid[1,:] = [u₀, u₁]
		for j ∈ 1:J
			pyramid[j+1,:] = blur_op.(pyramid[j,:])
		end
	end
	
	# construct flow buffer to store and return intermediate flows
	if retflows
		flows = Zygote.Buffer(Matrix{typeof(u₀)}(undef, (J+1,πnet.W+1)))
	end

	# init variables
	# use similar to init with CuArrays of on CUDA
	M, N, _, B = size(pyramid[end, 1])
	v̄        = similar(u₀, eltype(u₀), (M, N, 2, B))
	dual_var = similar(u₀, eltype(u₀), (M÷πnet.s, N÷πnet.s, πnet.M, B))
	Zygote.ignore() do
		fill!(v̄,0)
		fill!(dual_var,0)
	end
	v = nothing

	for j ∈ J:-1:0
		u₀, u₁ = pyramid[j+1,:]
		for w ∈ 0:πnet.W
			ū₁ = (j==J && w==0) ? u₁ : warp_bilinear(u₁, stopgrad ? Zygote.dropgrad(v̄) : v̄)
			v, dual_var = πnet[w+1](u₀, ū₁, v̄, dual_var)
			v̄ = v
			if retflows
				flows[j+1,w+1] = v
			end
		end
		if j > 0
			v̄ = 2*upsample_bilinear(v̄, (2,2))
			dual_var = 2*upsample_bilinear(dual_var, (2,2))
		end
	end

	if retflows
		return copy(flows)
	end
	return unpad(v, preparams[2])
end

#=============================================================================
                                  Projection 
=============================================================================#

Π!(t::AbstractArray) = clamp!(t, 0, Inf)
function Π!(c::Union{Conv,ConvTranspose})
	c.weight ./= max.(1, sqrt.(sum(abs2, c.weight, dims=(1,2))))
	return nothing
end
function Π!(n::BCANet)
	Π!.((n.A..., n.Bᵀ..., n.τ..., n.λ...))
	return nothing
end
function Π!(πn::PiBCANet)
	Π!.(πn.net)
	return nothing
end

#=============================================================================
                                  SHOW 
=============================================================================#

function Base.show(io::IO, n::BCANet)
	print(io, "BCANet(K=", n.K, ", M=",n.M, ", P=", n.P, ", s=", n.s, ")")
	nps = sum(length, params(n)); nps_unit = ""
	if nps > 1000
		nps = nps ÷ 1000
		nps_unit = "k"
	end
	printstyled(io, "  # ", nps, nps_unit*" parameters"; color=:light_black)
end

function Base.show(io::IO, πn::PiBCANet)
	print(io, "PiBCANet(W=", πn.W, ", shared=", πn.shared, ", K=", πn.K, ", M=", πn.M, ", P=", πn.P, ", s=", πn.s, ")")
	nps = sum(length, params(πn.net[1]))
	if !πn.shared
		nps *= πn.W
	end
	nps_unit = ""
	if nps > 1000
		nps = nps ÷ 1000
		nps_unit = "k"
	end
	printstyled(io, "  # ", nps, nps_unit*" parameters"; color=:light_black)
end

