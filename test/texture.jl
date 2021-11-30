using Images, TestImages, ColorTypes, FixedPointNumbers
using Interpolations
using UnrolledFlowNetworks

ds = MPISintelDataset("dataset/MPI_Sintel"; split="val", gray=true)

#img = RGBA{N0f8}.(testimage("lighthouse"))
#img′ = reinterpret(NTuple{4,UInt8}, img)

# texturearray = CuTextureArray(img′)
# texture = CuTexture(texturearray; normalized_coordinates=true)
#

function backward_warp!(dst, texture, flow)
	

function warp(dst, texture, flow)
	tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	I = CartesianIndices(dst)
	a, b = params
	@inbounds if tid <= length(I)
		i,j = Tuple(I[tid])
		u,v = flow[i,j,:,1]
		x = i + u 
		y = j + v 
		dst[i,j] = texture[x,y]
	end
	return nothing
end

#outimg_d = CuArray{eltype(img′)}(undef, size(img)...);

function configurator(kernel)
	config = launch_configuration(kernel.fun)
	threads = min(length(outimg_d), config.threads)
	blocks = cld(length(outimg_d), threads)
	return (threads=threads, blocks=blocks)
end

# kernel = @cuda launch=false warp(outimg_d, texture, (2,3))
# config = configurator(kernel)
# kernel(outimg_d, texture, (4,5); config...)
# 
# outimg = Array(outimg_d)
# save("imgwarp2.png", reinterpret(eltype(img), outimg))
# save("img.png", img)
