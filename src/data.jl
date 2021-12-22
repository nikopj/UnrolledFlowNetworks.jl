#=
data.jl
=#

"""
    loadargs(fn::String)

Load arguments (.yml) YAML file.
"""
loadargs(fn::String) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})

"""
    saveargs(fn::String, args::Dict{Symbol,Any})

Save arguments dictionary to (.yml) YAML file.
"""
saveargs(fn::String, args::Dict{Symbol,Any}) = YAML.write_file(fn, args)

#=============================================================================
                            Image <-> Tensor
=============================================================================#

"""
    tensor2img(A::Array{<:Real,2})

Convert matrix of reals into grayscale image.
"""
function tensor2img(A::Array{<:Real,2})
	tensor2img(Gray, A)
end

"""
    tensor2img(A::Array{<:Real,4})

Convert multi-dim tensor to (batched, if size(A,4)>1) matrix of 
Gray (size(A,3)==1) or RGB (size(A,3)==3) values.
"""
function tensor2img(A::Array{<:Real,4})
	if size(A)[3] == 1
		return cat([tensor2img(A[:,:,1,i]) for i ∈ 1:size(A,4)]..., dims=3)
	end
	out = cat([tensor2img(RGB, permutedims(A[:,:,:,i], (3,1,2))) for i ∈ 1:size(A,4)]..., dims=3)
	if size(out, 3) == 1
		return out[:,:,1]
	end
	return out
end

"""
    tensor2img(ctype::Type, A::Array{<:Real})

Convert multi-dim tensor of real values into image with elements of ctype{N0f8}.
"""
function tensor2img(ctype::Type, A::Array{<:Real}) 
	reinterpret(reshape, ctype{N0f8}, N0f8.(clamp.(A,0,1)))
end

"""
    img2tensor(T::Type, img)

Convert grayscale or RGB image to 4-tensor (W,H,C,1) with elements of type T
(default T==Float32).
"""
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

"""
    tensorload(T::Type, path::String; gray::Bool=false)

Load image or flow (.flo) file into 4D tensor. Optionally convert RGB images to
grayscale. Default type is Float32.
"""
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

"""
    flo2tensor(T::Type, flo)

Convert .flo file loaded from OpticalFlowUtils.jl to 4D tensor (W,H,2,1).
"""
function flo2tensor(T::Type, flo)
	return permutedims(convert(Array{T,3},flo), (2,3,1)) |> Flux.unsqueeze(4)
end
flo2tensor(A) = flo2tensor(Float32, A)

"""
    colorflow(flow::Array{T,4}, maxflow)

Convert flow tensor to HSV colorspace for display. Flows are normalized (in norm) by maxflow.
"""
function colorflow(flow::Array{T,4}; maxflow=maximum(mapslices(norm, flow, dims=3))) where {T}
	CT = HSV{Float32}
	color(x1, x2) = ismissing(x1) || ismissing(x2) ?
		CT(0, 1, 0) :
		CT(180f0/π * atan(x1, x2), norm((x1, x2)) / maxflow, 1)
	x1 = selectdim(flow, 3, 1)
	x2 = selectdim(flow, 3, 2)
	out = color.(x1, x2)
	if size(out, 3) == 1
		return out[:,:,1]
	end
	return out
end

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

Base.getindex(ds::AbstractDataset, i::AbstractVector{Int}) = map(x->ds[x], i)

function Base.show(io::IO, ds::AbstractDataset)
	print(io, "Dataset(name=\"",ds.name,"\", length=",length(ds))
	print(io, ", eltype=",typeof(ds[1]),")")
end

#==============================================================================
                           FlyingChairs DATASET
==============================================================================#

mutable struct FlyingChairsDataset <: AbstractDataset
	name::String
	root::String
	data::Vector
	gray::Bool
end

function FlyingChairsDataset(root::String; split="trn", gray=false)
	# get filenames of flows
	vecfn = filter(x->occursin(".flo", x), readdir("dataset/FlyingChairs/data/"))
	# get numbers within filenames
	vecindex = map(x->parse(Int, SubString(x, 1:5)), vecfn)

	# get training or val subset
	if split ∈ ("trn", "val")
		trn_val = readdlm(joinpath(root, "train_val.txt"), Int)[:,1] .== (split=="trn" ? 1 : 2)
		vecindex = vecindex[trn_val]
	end

	# function to get filenames for "img1", "img2", or "flow"
	getfilenames(name) = begin
		ext = name == "flow" ? ".flo" : ".ppm"
		files = map(x->joinpath(root, @sprintf("data/%05d_", x)*name*ext), vecindex)
	end
	files = getfilenames.(("img1", "img2", "flow"))

	# data as vector of named tuples of file names
	data = [(frame0=files[1][i], frame1=files[2][i], flow=files[3][i]) for i ∈ 1:length(files[3])]

	return FlyingChairsDataset("FlyingChairs-"*split, root, data, gray)
end

Base.length(ds::FlyingChairsDataset) = length(ds.data)
function Base.getindex(ds::FlyingChairsDataset, i::Int)
	d = ds.data[i]
	F = FlowSample(tensorload(d.frame0, gray=ds.gray), 
	               tensorload(d.frame1, gray=ds.gray), 
	               tensorload(d.flow),
	               missing)
	F.masks = ones(eltype(F.flows), size(F.frame0)[1:2]..., 1, 1)
	return F
end

#==============================================================================
                            MPI-Sintel DATASET
==============================================================================#


mutable struct MPISintelDataset <: AbstractDataset
	name::String
	root::String
	data::FlowData
	findex::Vector{Tuple{Int,Int}}
	gray::Bool
end

function MPISintelDataset(root::String; split="trn", type="clean", gray=false)
	if split == "all"
		d = readdir(joinpath(root, "training/clean"))
	else
		d = joinpath(root, "training", split*".txt") |> readdlm 
	end
	loadset(dir) = begin
		paths = d .|> x->joinpath(root, "training", dir, x)
		listfn = [readdir(p, join=true) for p in paths]
		N = sum(length.(listfn))
		P = meter.Progress(N, desc=uppercase(split)*"-"*dir*" ")
		[ntuple(i-> begin meter.next!(P); vecfn[i]; end, length(vecfn)) for vecfn in listfn]
	end
	data = FlowData(loadset.((type,"flow","occlusions","invalid"))...)
	findex = []
	for i=1:length(data.flow), j=1:length(data.flow[i])
		push!(findex, (i,j))
	end
	return MPISintelDataset("MPI-Sintel-"*split, root, data, findex, gray)
end

Base.length(ds::MPISintelDataset) = length(ds.findex)
function Base.getindex(ds::MPISintelDataset, i::Int)
	p, q = ds.findex[i]
	fD = ds.data
	FlowSample(tensorload(fD.frame[p][q], gray=ds.gray), 
	           tensorload(fD.frame[p][q+1], gray=ds.gray), 
	           tensorload(fD.flow[p][q]),
	           Float32.((1f0 .- tensorload(fD.invalid[p][q])).*(1f0 .- tensorload(fD.occlusion[p][q]))))
end

#==============================================================================
                                 DATALOADER
==============================================================================#

mutable struct Dataloader
	dataset::AbstractDataset
	transform::Function        
	batch_size::Int
	minibatches::AbstractVector # shuffled vector of mini-batch indices in dataset
end

function Dataloader(ds::AbstractDataset, training::Bool; batch_size::Int=1, crop_size::Int=128, scale=0, J=0, σ::Union{<:Real,Tuple,Vector}=0, device=identity)
	σ′ = training ? Float32.((scale+1) .* σ./255) : 0f0
	faugment(F) = training ? augment(F, crop_size) : F
	blur_ops = (ConvGaussian(groups=size(ds[1].frame0,3), stride=2, device=device),
		ConvGaussian(groups=2, stride=2, device=device),
		ConvGaussian(groups=1, stride=2, device=device))
	xfrm(Fb) = transform(faugment, Fb, σ′, scale, J, blur_ops; device=device)
	dl = Dataloader(ds, xfrm, batch_size, 1:length(ds))
	shuffle!(dl)
end

function Dataloader(ds::AbstractDataset, transform::Function, bs::Int)
	dl = Dataloader(ds, transform, bs, 1:length(ds))
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
	# pad = calcpad(size(Fb.frame0)[1:2], 2^(scale+J))
	# Fb = broadcast!(x->pad_reflect(x, pad, dims=(1,2)), Fb)

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
	i, j = rand(1:M-cs+1), rand(1:N-cs+1)
	crop(x, cs, (i,j))
end
function randcrop(F::FlowSample, cs::Int)
	M, N = size(F)[1:2]
	i, j = rand(1:M-cs+1), rand(1:N-cs+1)
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

