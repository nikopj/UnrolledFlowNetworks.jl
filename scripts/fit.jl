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
ds_trn = TheDataset(args[:data][:root]; split="trn", args[:data][:ds_params]...)
ds_val = TheDataset(args[:data][:root]; split="val", args[:data][:ds_params]...)
ds_tst = MPISintelDataset("dataset/MPI_Sintel"; split="all", gray=args[:data][:ds_params][:gray])

# ensure nosie-level range is given as tuple
if args[:data][:dl_params][:σ] isa Vector
	args[:data][:dl_params][:σ] = tuple(args[:data][:dl_params][:σ]...)
end

# build dataloaders
dl_trn = Dataloader(ds_trn, true; args[:data][:dl_params]..., device=device)
dl_val = Dataloader(ds_val, false; batch_size=1, J=args[:data][:dl_params][:J], scale=args[:data][:dl_params][:scale], device=device)
dl_tst = Dataloader(ds_tst, false; batch_size=1, J=args[:data][:dl_params][:J], scale=args[:data][:dl_params][:scale], device=device)
loaders = (trn=dl_trn, val=dl_val, tst=dl_tst)
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

lossfcn = args[:train][:Loss]
if occursin("L1", lossfcn)
	args[:train][:Loss] = L1Loss
elseif occursin("EPE", lossfcn)
	args[:train][:Loss] = EPELoss
else
	@error "Dataset $lossfcn not implemented."
end

train!(net, loaders, opt; start=start, device=device, args[:train]...)

