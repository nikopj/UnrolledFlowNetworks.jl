#==============================================================================
                                 SAMPLES
==============================================================================#
const FlowSample = NamedTuple{(:img1, :img2, :flow, :mask), NTuple{4,AbstractArray}}

struct FlowFile
	img1::String
	img2::String
	flow::String
	mask::Union{Missing,String}
	occlusion::Union{Missing,String}
	invalid::Union{Missing,String}
end

function augment(fs::FlowSample; cropsize=nothing)
	if !isnothing(cropsize)
		# random crop
		M, N = size(fs.img1)[1:2]
		start = rand.((1:M-cropsize[1]+1, 1:N-cropsize[2]+1))
		img1 = crop(fs.img1, cropsize, start) |> collect
		img2 = crop(fs.img2, cropsize, start) |> collect
		flow = crop(fs.flow, cropsize, start) |> collect
		mask = size(fs.mask)[1:2] == (M,N) ? crop(fs.mask, cropsize, start) |> collect : fs.mask
	end

	# random flip
	for dim ∈ (1,2)
		if rand() < 0.5
			reverse!.((img1, img2, flow, mask), dims=dim)
			selectdim(flow, 3, dim) .*= -one(eltype(flow))
		end
	end
	return FlowSample((img1, img2, flow, mask))
end
augment(; kws...) = x->augment(x; kws...)
