#=
data.jl
=#

loadargs(fn::String) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})
saveargs(fn::String, args::Dict{Symbol,Any}) = YAML.write_file(fn, args)

#=============================================================================
                            Image <-> Tensor
=============================================================================#

function tensor2img(A::Array{<:Real,2})
	tensor2img(Gray, A)
end

function tensor2img(A::Array{<:Real,4})
	if size(A)[3] == 1
		return cat([tensor2img(A[:,:,1,i]) for i ∈ 1:size(A,4)]..., dims=3)
	end
	return cat([tensor2img(RGB, permutedims(A[:,:,:,i], (3,1,2))) for i ∈ 1:size(A,4)]..., dims=3)
end

function tensor2img(ctype, A::Array{<:Real}) 
	reinterpret(reshape, ctype{N0f8}, N0f8.(clamp.(A,0,1)))
end

function img2tensor(T::Type, img)
	B = T.(reinterpret(reshape, N0f8, img) |> collect)
	if ndims(B) == 3
		B = permutedims(B, (2,3,1))
		B = reshape(B, size(B)..., 1)
	elseif ndims(B) == 2
		B = reshape(B, size(B)..., 1, 1)
	end
	return B
end
img2tensor(A) = img2tensor(Float32, A)

function flo2tensor(T::Type, flo)
	return permutedims(convert(Array{T,3},flo), (2,3,1)) |> Flux.unsqueeze(4)
end
flo2tensor(A) = flo2tensor(Float32, A)

function tensorload(T::Type, path::String; gray::Bool=false)
	A = load(path)
	# load optical flow
	if occursin(".flo", path)
		return flo2tensor(A)
	end
	# load image
	return img2tensor(T, gray ? Gray.(A) : A)
end
tensorload(path::String; gray::Bool=false) = tensorload(Float32, path; gray=gray)

#==============================================================================
                                 FLOWDATA/SAMPLE
==============================================================================#

struct FlowData
	frame
	flow
	occlusion
	invalid
end

mutable struct FlowSample
	frame0 
	frame1 
	flows  
	masks  
end
Base.size(F::FlowSample) = size(F.frame0)

function broadcast!(f, F::FlowSample)
	for name ∈ fieldnames(FlowSample)
		setproperty!(F, name, getproperty(F, name) |> f)
	end
	return F
end

Flux.cpu(F::FlowSample) = broadcast!(Flux.cpu, F)
Flux.gpu(F::FlowSample) = broadcast!(Flux.gpu, F)

function Base.cat(Fb::FlowSample...; dims=4)
	FlowSample(cat([F.frame0 for F ∈ Fb]..., dims=dims),
	           cat([F.frame1 for F ∈ Fb]..., dims=dims),
	           cat([F.flows for  F ∈ Fb]..., dims=dims),
	           cat([F.masks for  F ∈ Fb]..., dims=dims))
end

function Base.show(io::IO, F::FlowSample)
	print(io, "FlowSample((frame0,frame1,flows,masks), size=",size(F.frame0))
	if F.flows isa Tuple || F.flows isa Vector
		print(io, ", J=", length(F.flows)-1)
	end
	print(io, ")")
end

#==============================================================================
                                 DATASET
==============================================================================#

abstract type AbstractDataset end

struct MPISintelDataset <: AbstractDataset
	name::String
	root::String
	data::FlowData
	findex::Vector{Tuple{Int,Int}}
end

function MPISintelDataset(root::String; split="trn", gray=false)
	d = joinpath(root, "training", split*".txt") |> readdlm 
	f(x) = gray ? begin
		x = load(x)
		x = eltype(x) <: RGB ? Gray.(x) : x
	end : load(x)
	loadset(dir) = begin
		paths = d .|> x->joinpath(root, "training", dir, x)
		listfn = [readdir(p, join=true) for p in paths]
		N = sum(length.(listfn))
		P = meter.Progress(N, desc=uppercase(split)*"-"*dir*" ")
		[ntuple(i-> begin meter.next!(P); f(vecfn[i]); end, length(vecfn)) for vecfn in listfn]
	end
	data = FlowData(loadset.(("clean","flow","occlusions","invalid"))...)
	findex = []
	for i=1:length(data.flow), j=1:length(data.flow[i])
		push!(findex, (i,j))
	end
	return MPISintelDataset("MPI-Sintel-"*split, root, data, findex)
end

Base.length(ds::MPISintelDataset) = length(ds.findex)
function Base.getindex(ds::MPISintelDataset, i::Int)
	p, q = ds.findex[i]
	fD = ds.data
	FlowSample(img2tensor(fD.frame[p][q]), 
	           img2tensor(fD.frame[p][q+1]), 
	           flo2tensor(fD.flow[p][q]),
	           Float32.((1f0 .- fD.invalid[p][q]).*(1f0 .- fD.occlusion[p][q])))
end
Base.getindex(ds::MPISintelDataset, i::AbstractVector{Int}) = i .|> x-> ds[x]

function Base.show(io::IO, ds::MPISintelDataset)
	print(io, "MPISintelDataset(name=\"",ds.name,"\", length=",length(ds))
	print(io, ", eltype=",typeof(ds[1]),")")
end

#==============================================================================
                                 DATALOADER
==============================================================================#

function getMPISintelLoaders(root::String; gray=false, batch_size=10, crop_size=128, σ=1, scale=0, J=0, device=identity)
	@warn "NOT LOADING TRAINSET. UPDATE BEFORE SUBMITTING JOBS"
	ds_trn = MPISintelDataset(root; split="trn", gray=gray)
	ds_val = MPISintelDataset(root; split="val", gray=gray)
	dl_trn = Dataloader(ds_val, true; batch_size=batch_size, crop_size=crop_size, scale=scale, J=J, σ=σ, device=device)
	dl_val = Dataloader(ds_val, false; batch_size=1, scale=scale, J=J, σ=0, device=device)
	return (trn=dl_trn, val=dl_val, tst=dl_val)
end

""" Dataloader
"""
mutable struct Dataloader
	dataset::AbstractDataset
	transform::Function        
	batch_size::Int
	minibatches::AbstractVector # shuffled vector of mini-batch indices in dataset
end

function Dataloader(ds::AbstractDataset, transform::Function, bs::Int)
	dl = Dataloader(ds, transform, bs, 1:length(ds))
	shuffle!(dl)
end

function Dataloader(ds::MPISintelDataset, training::Bool; batch_size::Int=1, crop_size::Int=128, scale=0, J=0, σ::Union{<:Real,Tuple,Vector}=0, device=identity)
	σ′ = Float32.((scale+1) .* σ./255)
	faugment(F) = training ? augment(F, crop_size) : F
	blur_ops = (ConvGaussian(1; groups=size(ds[1].frame0,3), stride=2, device=device),
		ConvGaussian(1; groups=2, stride=2, device=device),
		ConvGaussian(1; groups=1, stride=2, device=device))
	xfrm(Fb) = transform(faugment, Fb, σ′, scale, J, blur_ops; device=device)
	dl = Dataloader(ds, xfrm, batch_size, 1:length(ds))
	shuffle!(dl)
end

Base.length(dl::Dataloader) = length(dl.minibatches)

function Base.getindex(dl::Dataloader, i::Int)
	1 <= i <= length(dl) || throw(BoundsError(dl, i))
	dl.transform(dl.dataset[dl.minibatches[i]]) 
end

function Base.iterate(dl::Dataloader, state::Int=1)
	if state > length(dl)
		shuffle!(dl)      # shuffle minibatches at end of epoch
		return nothing
	end
	return (dl[state], state+1)
end

function shuffle!(dl::Dataloader)
	dl.minibatches = (collect∘partition)(shuffle(1:length(dl.dataset)), dl.batch_size)
	if length(dl.minibatches[end]) != dl.batch_size
		pop!(dl.minibatches)
	end
	return dl
end

function Base.show(io::IO, dl::Dataloader)
	print(io, "Dataloader(dataset=",dl.dataset,",\nbatch_size=",dl.batch_size)
	print(io, ", length=",length(dl))
	print(io, ", eltype=",typeof(dl[1]),")\n")
end

#==============================================================================
                                 TRANSFORMS
==============================================================================#
function augment(F::FlowSample, crop_size)
	F = randcrop(F, crop_size)
	F = randflip(F, 0.5, 1)
	F = randflip(F, 0.5, 2)
	return F
end

function transform(f_augment::Function, Fb::Vector{FlowSample}, σ, scale, J, blur_ops; device=identity) 
	# batch
	Fb = cat(f_augment.(Fb)..., dims=4)

	# add noise
	Fb.frame0 = awgn(Fb.frame0, σ)[1] |> x->clamp!(x,0,1)
	Fb.frame1 = awgn(Fb.frame1, σ)[1] |> x->clamp!(x,0,1)

	# pad
	pad = calcpad(size(Fb.frame0)[1:2], 2^(scale+J))
	Fb = broadcast!(x->pad_reflect(x, pad, dims=(1,2)), Fb)

	Fb = Fb |> device

	# blur to scale
	for i ∈ 1:scale
		Fb.frame0 = blur_ops[1](Fb.frame0)
		Fb.frame1 = blur_ops[1](Fb.frame1)
		Fb.flows  = blur_ops[2](Fb.flows) ./ 2f0
		Fb.masks  = blur_ops[3](Fb.masks)
	end

	# create gaussian pyramid
	frame0 = Fb.frame0
	frame1 = Fb.frame1
	flows = [Fb.flows]
	masks = [Fb.masks]
	for j ∈ 1:J
		push!(flows, blur_ops[2](flows[j]) ./ 2f0)
		push!(masks, blur_ops[3](masks[j]))
	end

	return FlowSample(frame0, frame1, flows, masks)
end

#==============================================================================
                                 OPERATIONS
==============================================================================#

flip(x, dim=1) = reverse(x, dims=dim)
randflip(x::AbstractArray, p::Real, dim=1) = rand() < p ? flip(x,dim) : x
function randflip(F::FlowSample, p::Real, dim=1)
	rand() < p ? F : begin
		flows = flip(F.flows, dim)
		selectdim(flows, 3, dim) .*= -1
		FlowSample(flip(F.frame0,dim), flip(F.frame1,dim), flows, flip(F.masks,dim))
	end
end

function crop(x::AbstractArray, cs::Int, ij::Tuple{Int,Int})
	i,j = ij
	selectdim( selectdim(x, 1, i:i+cs-1), 2, j:j+cs-1 )
end
function randcrop(x::AbstractArray, cs::Int)
	M, N = size(x)[1:2]
	i, j = rand(1:M-cs), rand(1:N-cs)
	crop(x, cs, (i,j))
end
function randcrop(F::FlowSample, cs::Int)
	M, N = size(F)[1:2]
	i, j = rand(1:M-cs), rand(1:N-cs)
	C(x) = crop(x, cs, (i,j))
	FlowSample(C.((F.frame0, F.frame1, F.flows, F.masks))...)
end

"""
    awgn(x::Union{AbstractArray,CuArray}, σ)

Additive Gaussian White Noise with noise-level σ.
"""
awgn(x::AbstractArray, σ::Union{AbstractArray,<:Real}) = x .+ σ.*randn(eltype(x), size(x)...), σ
awgn(x::Union{CuArray, SubArray{<:Any,<:Any,<:CuArray}}, σ::Union{AbstractArray,<:Real}) = x .+ σ.*CUDA.randn(eltype(x), size(x)...), σ

"""
    awgn(x, σ::Vector)

AWGN where noise-levels corresponding to each batch element are supplied.
"""
function awgn(x::Union{T, SubArray{<:Any,<:Any,T}}, σ::Vector) where {T<:Union{AbstractArray,CuArray}}
	@assert length(σ) == size(x, ndims(x)) "Noise-levels length must match batch-size"
	awgn(x, reshape(σ, ones(Int, ndims(x)-1)..., :))
end

"""
    awgn(x, (σ₁, σ₂))

AWGN over batched tensor x, where noise level is drawn independently and uniformly 
between (σ₁, σ₂) for each batch element of x.
"""
function awgn(x, I::NTuple{2,<:Number})
	@assert I[2] ≥ I[1] "I[2] ≱ I[1] ($(I[2]) ≱ $(I[1]))"
	σ = I[1] .+ (I[2]-I[1]).*rand(eltype(x), size(x,ndims(x)))
	awgn(x, σ)
end

