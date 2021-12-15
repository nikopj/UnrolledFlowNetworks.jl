using UnrolledFlowNetworks, Flux, CUDA, FileIO
CUDA.allowscalar(false)

# get argument filename and device
if isinteractive()
	fn = "scripts/args.yml"
else
	fn = ARGS[1]
end
device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu

args = loadargs(fn)
println(args)
mkpath(args[:train][:savedir])

# instantiate network
loadingckpt = args[:ckpt] ≠ nothing && isfile(args[:ckpt])
net = PiBCANet(; args[:net]..., init=!loadingckpt) |> device
@show net

# instantiate optimizer
opt = ADAM(args[:opt][:η]) |> device

# load data
@warn "Only using VAL set. Change before submitting jobs."
ds_trn = MPISintelDataset(args[:data][:root], split="trn", gray=args[:data][:gray])
ds_val = MPISintelDataset(args[:data][:root], split="val", gray=args[:data][:gray])

# build dataloaders
dl_trn = Dataloader(ds_trn, true; args[:data][:params]..., device=device)
dl_val = Dataloader(ds_val, false; batch_size=1, J=args[:data][:params][:J], scale=args[:data][:params][:scale], device=device)
loaders = (trn=dl_trn, val=dl_val, tst=dl_val)
@show loaders

start = 1
if loadingckpt
	@info "Loading checkpoint $(args[:ckpt])..."
	ckpt = load(args[:ckpt])
	net     = ckpt[:net] |> device
	start   = ckpt[:epoch] + 1
	opt.eta = ckpt[:η]
end

# save updated argument file
args[:ckpt] = joinpath(args[:train][:savedir], "net.bson")
fn = joinpath(args[:train][:savedir], "args.yml")
saveargs(fn, args)

train!(net, loaders, opt; start=start, device=device, args[:train]...)
 
