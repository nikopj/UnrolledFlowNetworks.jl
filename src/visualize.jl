#=
visualize.jl
=#

"""
  dictionary(m::CDLNet)

Returns the network's synthesis dictionary filters (m.D.weight) in a
gray-scale mosaicview grid, ordered (col-major) by their variance.  
Filters are normalized globally. The filter permutation of the grid is 
also returned.
"""
function dictionary(m::CDLNet)
	# normalize dictionary and get magnitude spectrum
	D = mapslices(x->x./norm(x), copy(m.D.weight), dims=(1,2))
	Df = fft(pad_constant(D, (0,128-size(D,1),0,128-size(D,2)), dims=(1,2)) , (1,2)) |> x->fftshift(x, (1,2)) .|> abs

	# sort dictionary
	p = sortperm(vec(var(D; dims=(1,2))))
	nrow, ncol = ceil(Int,sqrt(m.M)), ceil(Int,m.M / nrow)

	# sort and mosaic dictionary to matrix
	D = mosaicview(D[:,:,1,p],  npad=2, nrow=nrow)
	Df= mosaicview(Df[:,:,1,p], npad=2, nrow=nrow)
	if nrow*ncol > m.M                                  # put order in grid form
		p = [p; -ones(Int, nrow*ncol - m.M)]
	end
	Π(x) = (x .- minimum(x))./(maximum(x) - minimum(x)) # map to [0,1]
	return Gray.(Π(D)), Gray.(Π(Df)), reshape(p, nrow, ncol)
end

function passthrough(m::CDLNet, x::Array{<:Real,4}, σ::Real)
	Π(x) = (x .- minimum(x))./(maximum(x) - minimum(x))
	y = awgn(x, σ)
	x̂, z = m(y; σ=σ)
end

function passthrough(m::CDLNet, x::Array{<:Real,N}, σ::Real) where N
	@assert 2 ≤ N ≤ 3
	x = reshape(x, size(x)..., ones(Int, 4-N)...)
	x̂, z = passthrough(m, x, σ)
	return (collect ∘ reshape)(x̂, size(x)[1:N]), z
end

function passthrough(m::CDLNet, x::AbstractMatrix, σ::Real)
	x̂, z = passthrough(m, img2tensor(x), σ)
	return tensor2img(x̂), z
end



