module Data

import DataLoaders.LearnBase: getobs, nobs
using DelimitedFiles
import LearnBase
using DataFrames
using OpticalFlowUtils
using Images
using FileIO
using LinearAlgebra, Statistics
using CUDA

include("tensors.jl")
export tensor2img, img2tensor, tensorload
export FlyingChairsDataSet, augment

include("samples.jl")
include("operations.jl")

#==============================================================================
                                 DATASET
==============================================================================#

abstract type TensorDataSet end

LearnBase.nobs(ds::TensorDataSet) = length(ds.files)
LearnBase.getobs(ds::TensorDataSet, i::Int) = tensorload(ds.files[i]; gray=ds.gray) |> ds.transform
LearnBase.getobs(ds::TensorDataSet, i::AbstractVector{Int}) = map(x->getobs(ds,x), i)

struct FlowDataSet <: TensorDataSet
	name::String
	files::Vector{FlowFile}
	transform::Function
	gray::Bool
end

function FlyingChairsDataSet(root::String, transform=identity; 
                             gray=false,
                             split=:all,
                             flomin=0,
                             flomax=Inf)
	# trainval split indices
	trainval = readdlm(joinpath(root, "train_val.txt"), Int)[:,1]
	if split ∈ (:trn, :val)
		keep =  trainval .== (split==:trn ? 1 : 2)
	else
		keep = ones(length(trainval))
	end

	# only keep samples with flow magnitude in [flowmin, flomax]
	stats = joinpath(root, "stats.csv") |> load |> DataFrame
	keep = keep .* (stats.min .≥ flomin) .* (stats.max .≤ flomax) .|> Bool

	files = begin
		fn = readdir(abspath(joinpath(root, "data")); join=true)
		fn_img1 = filter(x->occursin("img1",x), fn)
		fn_img2 = filter(x->occursin("img2",x), fn)
		fn_flow = filter(x->occursin("flow",x), fn)
		ffiles = FlowFile.(fn_img1, fn_img2, fn_flow, missing, missing, missing)
	end
	return FlowDataSet("FlyingChairs-$(split)_$(flomin)_$(flomax)", files[keep], transform, gray)
end

function tensorload(T::Type, fs::FlowFile; gray=false)
	img1, img2, flow = tensorload.((fs.img1, fs.img2, fs.flow); gray=gray)
	mask = ismissing(fs.mask) ? ones(Bool, 1, 1, 1) : tensorload(Bool, fs.mask)

	occlusion = ismissing(fs.occlusion) ? missing : tensorload(Bool, fs.occlusion) 
	mask = ismissing(occlusion) ? mask : mask .* .!occlusion

	invalid = ismissing(fs.invalid) ? missing : tensorload(Bool, fs.invalid) 
	mask = ismissing(invalid) ? mask : mask .* .!invalid
	return FlowSample((img1, img2, flow, T.(mask)))
end
tensorload(fs::FlowFile; gray=false) = tensorload(Float32, fs; gray=gray)

end
