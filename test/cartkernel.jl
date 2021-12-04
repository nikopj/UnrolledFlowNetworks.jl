using CUDA

function cartesian2_kernel!(dst, max_idx, s1, s2)
	index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

	@inbounds if index <= max_idx
		dst[index] = CartesianIndex(s1[index], s2[index])
	end
	return nothing
end

function cartesian2(dst::CuArray{CartesianIndex{2}}, sources...)
	max_idx = length(dst)
	args = (dst, max_idx, sources...)
	kernel = @cuda launch=false cartesian2_kernel!(args...)
	config = launch_configuration(kernel.fun; max_threads=256)
	threads = min(max_idx, config.threads)
	blocks = cld(max_idx, threads)
	kernel(args...; threads=threads, blocks=blocks)
	return dst
end
