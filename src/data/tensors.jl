#=============================================================================
                            Image <-> Tensor
=============================================================================#

function tensorload(T::Type, file::String; gray=false) 
	if occursin(".flo", file)
		return flo2tensor(T, load(file))
	end
	img = gray ? Gray.(load(file)) : load(file)
	return img2tensor(T, img)
end
tensorload(file::String; gray=false) = tensorload(Float32, file; gray=gray)

function flo2tensor(T::Type, flow::AbstractArray)
	return permutedims(convert(Array{T,3},flow), (2,3,1)) 
end
flo2tensor(A) = flo2tensor(Float32, A)

function colorflow(flow::Array{T,3}; maxflow=maximum(sqrt.(sum(abs2, flow, dims=3)))) where {T}
	CT = HSV{Float32}
	color(x1, x2) = ismissing(x1) || ismissing(x2) ?
		CT(0, 1, 0) :
		CT(180f0/π * atan(x1, x2), norm((x1, x2)) / maxflow, 1)
	x1 = selectdim(flow, 3, 1)
	x2 = selectdim(flow, 3, 2)
	return color.(x1, x2)
end

"""
    img2tensor(T::Type, img::Matrix{<:Union{RGB,Gray})

Returns tensor of type T with 3 dimensions (HWC) from RGB or Gray image.
See Images.jl's [color conversions guide](https://juliaimages.org/latest/tutorials/quickstart/#Color-conversions-are-construction/view)
for internals.
"""
function img2tensor(T::Type, img::Matrix{<:RGB})
	img_CHW = channelview(img)
	img_HWC = permutedims(img_CHW, (2,3,1))
	return T.(img_HWC)
end
img2tensor(T::Type, img::Matrix{<:Gray}) = reshape(T.(img), size(img)..., 1)
img2tensor(img::Matrix{<:Union{RGB,Gray}}) = img2tensor(Float32, img)

"""
    tensor2img(A::Array{T,3}) 

Convert HWC tensors to 8bit Gray or RGB images.
Values are clamped to [0,1].
"""
function tensor2img(A::Array{T,3}) where T
	A = clamp.(A, 0, 1)
	if size(A,3) == 1
		return Gray{N0f8}.(A[:,:,1])
	elseif size(A,3) == 2
		return colorflow(A)
	elseif size(A,3) == 3
		img_CHW = permutedims(A, (3,1,2))
		return reinterpret(reshape, RGB{N0f8}, N0f8.(img_CHW))
	else
		@error "tensor2img: size(A,3) = $(size(A,3)) ∉ {1,3}."
	end
end

