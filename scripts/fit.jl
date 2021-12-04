using UnrolledFlowNetworks, Flux, CUDA, FileIO
CUDA.allowscalar(false)

# get argument filename and device
#fn = ARGS[1]
fn = "scripts/args.yml"
device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu

args = loadargs(fn)
mkpath(args[:train][:savedir])

# instantiate network
loadingckpt = args[:ckpt] ≠ nothing && isfile(args[:ckpt])
net = PiBCANet(; args[:net]..., init=!loadingckpt) |> device
@show net

# instantiate optimizer, load data
opt = ADAM(args[:opt][:η]) |> device
# @warn "NOT LOADING DATA. FIX BEFORE SUBMITTING JOBS!"
loaders = getMPISintelLoaders(args[:data][:root]; args[:data][:params]..., device=device)
# @show loaders

start = 1
if loadingckpt
	@info "Loading checkpoint $(args[:ckpt])..."
	ckpt = load(args[:ckpt])
	net     = ckpt[:net] |> device
	start   = ckpt[:epoch] + 1
	opt.eta = ckpt[:η]
end

train!(net, loaders, opt; start=start, device=device, args[:train]...)
 
# save updated argument file
@warn "NOT SAVING CKPT. FIX BEFORE SUBMITTING JOBS!"
# args[:ckpt] = joinpath(args[:train][:savedir], "net.bson")
# saveargs(fn, args)

