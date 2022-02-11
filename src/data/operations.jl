#==============================================================================
                                 OPERATIONS
==============================================================================#

function crop(x::AbstractArray, crop_size, start) 
	m, n = crop_size
	i, j = start
	return selectdim(selectdim(x, 1, i:i+m-1), 2, j:j+n-1)
end
crop(x, cs) = crop(x, cs, max.(1, (size(x)[1:2] .-1).÷2))

function randcrop(x::AbstractArray, crop_size::Tuple{Int,Int})
	M, N = size(x)[1:2]
	m, n = crop_size
	start = rand.((1:M-m+1, 1:N-n+1))
	return crop(x, crop_size, start)
end
randcrop(x, cs::Int) = randcrop(x, (cs,cs))

flip(x, dim) = reverse(x, dims=dim)
randflip(x::AbstractArray, dim::Int, p::Real) = rand() < p ? flip(x,dim) : x

