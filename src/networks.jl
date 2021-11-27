#=
networks.jl
=#

struct BCANet{K,C,Cᵀ,T,S,R}
	Aᵀ::NTuple{K,Cᵀ} # synthesis conv
	B::NTuple{K,C}   # analysis conv
	τ::S             # primal step-size
	λ::NTuple{K,T}   # lagrange multiplier
	K::Int           # iterations
	M::Int           # number of filters / 2
	P::Int           # filter size (square side-length)
	s::Int           # conv stride
	∇::R             # spatial gradient operator
end

Flux.@functor BCANet
Flux.trainable(n::BCANet) = (n.Aᵀ,n.B,n.λ,n.τ)

# INITIALIZATION
function BCANet(K::Int=10, M::Int=8, P::Int=7, s::Int=1, λ₀=1f-1; init=false)
	@assert K >1
	padl, padr = ceil(Int,(P-s)/2), floor(Int,(P-s)/2)
	pad = (padl, padr, padl, padr)
	W = randn(Float32, P, P, 1, 2*M)
	Aᵀ= ntuple(i->ConvTranspose(copy(W), false; pad=pad, stride=s, groups=2), K)
	B = ntuple(i->Conv(copy(W), false; pad=pad, stride=s, groups=2), K)
	if init
		L = powermethod(x->Aᵀ[1](B[1](x)), randn(Float32,128,128,2,1), maxit=500, tol=1e-2, verbose=false)[1]
		if L < 0
			println("ERROR: BCANet: powermethod: L<0. Something is very very wrong...")
		end
		@show L
		for k ∈ 1:K
			Aᵀ[k].weight ./= sqrt(L)
			B[k].weight  ./= sqrt(L)
		end
	end
	τ = ntuple(i->1f0*ones(Float32,1,1,1,1), K+1)
	λ = ntuple(i->Float32(λ₀)*ones(Float32,1,1,2*M,1), K)
	∇ = Conv(sobelkernel()[1], false; pad=1)
	BCANet(Aᵀ, B, τ, λ, K, M, P, s, ∇)
end

# Unrolled TVL1-BCA 
function (n::BCANet)(I₀,I₁,v̄=zero(Float32))
	I₀ᵖ, I₁ᵖ, p = preprocess(I₀, I₁, n.s)
	∇I = n.∇(I₁ᵖ)
	b = I₁ᵖ .- I₀ᵖ .- sum(∇I.*v̄, dims=3)
	α = sum(abs2, ∇I, dims=3) .+ 1f-7
	# k = 1, primal update
	vᵏ = zero(Float32); w = zero(Float32)
	v = ∇I.*(ST(b,n.τ[n.K+1].*α) .- b)./α
	for k ∈ 1:n.K
		# dual update
		y = w .+ n.B[k](2v .- vᵏ)
		q = sqrt.(y[:,:,1:n.M,:].^2 .+ y[:,:,n.M+1:end,:].^2)
		w = y ./ max.(1, cat(q,q,dims=3) ./ n.λ[k])
		# primal update
		vᵏ= v
		x = v .- n.τ[k].*n.Aᵀ[k](w)
		r = sum(abs2, ∇I.*x, dims=3) .+ b
		v = x .+ ∇I.*(ST(r, n.τ[k].*α) .- r)./α
	end
	return unpad(v, p[2])
end

# PROJECTION 
Π!(t::AbstractArray) = clamp!(t, 0, Inf)
function Π!(c::Union{Conv,ConvTranspose})
	c.weight ./= max.(1, sqrt.(sum(abs2, c.weight, dims=(1,2))))
end
function Π!(n::BCANet)
	Π!.((n.Aᵀ..., n.B..., n.τ..., n.λ...))
	return n
end

function Base.show(io::IO, n::BCANet)
	print(io, "BCANet(K=", n.K, ", M=",n.M, ", P=", n.P, ", s=", n.s)
	print(io, ")")
	nps = sum(length, params(n)); nps_unit = ""
	if nps > 1000
		nps = nps ÷ 1000
		nps_unit = "k"
	end
	printstyled(io, "  # ", nps, nps_unit*" parameters"; color=:light_black)
end

function recflowctf(net::BCANet, I₀, I₁, J::Int, W::Int; H=missing)
	if J == 0 && W==0
		v̄ = zero(Float32)
		v = net(I₀, I₁, v̄)
		return v, v̄
	end
	H = ismissing(H) ? ConvGaussian(1; stride=2) : H
	v⃗ = recflowctf(net, H(I₀), H(I₁), J-1, W, H=H) 
	v̄ = 2*upsample_bilinear(v⃗[1], (2,2))
	WI₁ = backward_warp(I₁, v̄)
	v = net(I₀, WI₁, v̄)
	return v, v⃗...
end

function flowctf(net::BCANet, I₀, I₁; J=0, W=0, retflows=false)
	# construct gaussian pyramid Δ
	H = ConvGaussian(1; stride=2)
	Δ = [(I₀,I₁)]

	for j ∈ 2:J+1
		push!(Δ, H.(Δ[j-1]))
	end

	# save intermediate flows
	if retflows; flows = []; end 

	# coarse to fine
	local WI₁, v
	v = zeros(Float32)
	for j ∈ J:-1:0
		for w ∈ 0:W
			WI₁ = j==J && w==0 ? Δ[j+1][2] : backward_warp(Δ[j+1][2], v)
			v = net(Δ[j+1][1], WI₁, v)
			if retflows; pushfirst!(flows,v); end
		end
		v = j>0 ? 2*upsample_bilinear(v, (2,2)) : v
	end
	if retflows
		return flows
	end 
	return v
end
