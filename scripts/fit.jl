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
net = PiBCANet(; args[:net]..., lipschitz_init=!loadingckpt) |> device
@show net

# instantiate optimizer
opt = ADAM(args[:opt][:η]) |> device

CUDA.memory_status()

# load data
dstype = args[:data][:dataset]
if dstype == "FlyingChairs"
	TheDataset = FlyingChairsDataset
elseif dstype == "MPI-Sintel"
	TheDataset = MPISintelDataset
else
	@error "Dataset $dstype not implemented."
end
# @warn "NOT LOADING DATASETS. Change before submitting jobs."
ds_trn = TheDataset(args[:data][:root], split="trn", gray=args[:data][:gray])
ds_val = TheDataset(args[:data][:root], split="val", gray=args[:data][:gray])

# build dataloaders
dl_trn = Dataloader(ds_trn, true; args[:data][:params]..., device=device)
dl_val = Dataloader(ds_val, false; batch_size=1, J=args[:data][:params][:J], scale=args[:data][:params][:scale], device=device)
loaders = (trn=dl_trn, val=dl_val, tst=dl_val)
@show loaders

CUDA.memory_status()

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

CUDA.memory_status()

train!(net, loaders, opt; start=start, device=device, args[:train]...)
 
