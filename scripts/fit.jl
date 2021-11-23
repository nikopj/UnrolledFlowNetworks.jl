using UnrolledFlowNetworks, Flux, CUDA, FileIO
CUDA.allowscalar(true)

function fit(fn; device=cpu)
	args = loadargs(fn)
	mkpath(args[:train][:savedir])
	loadingckpt = args[:ckpt] ≠ nothing && isfile(args[:ckpt])
	net = BCANet(args[:net]...; init=!loadingckpt) |> device
	@show net
	opt = ADAM(args[:opt][:η]) |> device
	loaders = getMPISintelLoaders(args[:data][:root]; args[:data][:params]...)
	@show loaders

	start = 1
	if loadingckpt
		@info "Loading checkpoint $(args[:ckpt])..."
		ckpt = load(args[:ckpt])
		net     = ckpt[:net] |> device
		start   = ckpt[:epoch] + 1
		opt.eta = ckpt[:η]
	end

	train!(net, loaders, opt; start=start, device=device, args[:train]...)
	args[:ckpt] = joinpath(args[:train][:savedir], "net.bson")
	saveargs(fn, args)
	return net, opt, loaders, args
end

if !isinteractive()
	using Pkg
	if isfile("Project.toml") && isfile("Manifest.toml")
		Pkg.activate(".")
	end

	fn = ARGS[1]
	device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu
	fit!(fn; device=device)
end

