using FileIO
using Base.Iterators: product
using UnrolledFlowNetworks: Utils
using .Utils: loadargs, saveargs, setrecursive!

loopargs = loadargs("scripts/loop_args.yml")
loop, args  = loopargs[:loop], loopargs[:args]
name, vnum  = loopargs[:name], loopargs[:version_num]

loopkeys = keys(loop) |> collect
println("Keys: ", loopkeys)

for (i, item) ∈ enumerate(product([loop[k] for k ∈ loopkeys]...))
	local fn

	for (j, it) ∈ enumerate(item)
		local k
		k = loopkeys[j]
		setrecursive!(args, k, it) && @warn "Key $k not found."
	end

	if args[:progressive]
		local k
		scales, warps = args[:net][:scales], args[:net][:warps]
		resolution = args[:train][:resolution]

		if warps isa Int
			warps = repeat([warps], scales)
		end

		k = 1
		prgargs = deepcopy(args)
		for j=1:scales, w=1:warps[j]
			# set network scales and warps 
			prgargs[:net][:scales] = j
			prgargs[:net][:warps]  = j>1 ? [w, [warps[l] for l in 1:j-1]...] : [w]

			# checkpoint is previous progressive model
			if j==1 && w==1
				prgargs[:ckpt] = nothing
			else
				prgargs[:ckpt] = prgargs[:train][:savedir] * "/net.bson"
			end
			@show (k,j,w) prgargs[:ckpt]

			# train scale j at proper resolution
			prgargs[:train][:resolution] = resolution + scales - j
			
			# naming
			version = name * "-$(vnum+i-1)-prg$k" 
			prgargs[:train][:savedir] = "models/" * version

			# save args
			println(version * ": ", item)
			fn = "args.d/" * version * ".yml"
			saveargs(fn, prgargs)
			k += 1
		end
	else
		# naming
		version = name * "-$(vnum+i-1)" 
		args[:train][:savedir] = "models/" * version

		# save args
		println(version * ": ", item)
		fn = "args.d/" * version * ".yml"
		saveargs(fn, args)
	end
end

