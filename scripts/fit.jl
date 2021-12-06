using UnrolledFlowNetworks, Flux, CUDA, FileIO
CUDA.allowscalar(false)

# get argument filename and device
# fn = ARGS[1]
fn = "scripts/args.yml"
device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu

args = loadargs(fn)
mkpath(args[:train][:savedir])

# instantiate network
loadingckpt = args[:ckpt] ≠ nothing && isfile(args[:ckpt])
net = PiBCANet(; args[:net]..., init=!loadingckpt) |> device
@show net

# instantiate optimizer
opt = ADAM(args[:opt][:η]) |> device

# load data
# ds_trn = MPISintelDataset(args[:data][:root], split="trn", gray=args[:data][:gray])
ds_val = MPISintelDataset(args[:data][:root], split="val", gray=args[:data][:gray])

# build dataloaders
dl_trn = Dataloader(ds_val, true; args[:data][:params]..., device=device)
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

train!(net, loaders, opt; start=start, device=device, args[:train]...)
 
# save updated argument file
args[:ckpt] = joinpath(args[:train][:savedir], "net.bson")
saveargs(fn, args)

