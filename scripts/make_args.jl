using FileIO
using Base.Iterators: product
using UnrolledFlowNetworks: loadargs, saveargs, setrecursive!

loopargs = loadargs("scripts/loop_args.yml")
loop, args     = loopargs[:loop], loopargs[:args]
name, version0 = loopargs[:name], loopargs[:version0]

loopkeys = keys(loop) |> collect
println("Keys: ", loopkeys)
for (i, item) ∈ enumerate(product([loop[k] for k ∈ loopkeys]...))
	local fn
	for (j, it) ∈ enumerate(item)
		k = loopkeys[j]
		setrecursive!(args, k, it) && @warn "Key $k not found."
	end
	version = name * "-$(version0+i-1)" 
	args[:train][:savedir] = "models/"*version
	println(version*": ", item)
	fn = "args.d/"*version*".yml"
	saveargs(fn, args)
end

