using UnrolledFlowNetworks
using Flux, CUDA
using FileIO

CUDA.allowscalar(false)
device = CUDA.functional() ? begin @info "Using GPU."; Flux.gpu end : identity

# get argument filename and device
if isinteractive()
	# fn = "scripts/args.yml"
	fn = "args.d/PiBCANet-prgtest-1-prg3.yml"
else
	fn = ARGS[1]
end

args = loadargs(fn)
println(args)
mkpath(args[:train][:savedir])

# instantiate network
loadingckpt = !isnothing(args[:ckpt]) && isfile(args[:ckpt])
net = PiBCANet(; args[:net]..., lipschitz_init=!loadingckpt) 
@show net

# instantiate optimizer
opt = ADAM(args[:opt][:η]) 

# load checkpoint
start = 1
if loadingckpt
	@info "Loading checkpoint $(args[:ckpt])..."
	ckpt = load(args[:ckpt])
	net′ = ckpt[:net] |> device
	print("checkpoint network: ")
	@show net′

	if net′ isa typeof(net)
		@info "ckpt-net is of same type: continuing training..."
		net = net′
		start   = ckpt[:epoch] + 1
		opt.eta = ckpt[:η]
	else
		@assert sum(net′.warps) == sum(net.warps) - 1 "ckpt-net must have 1 fewer BCANet modules than net!"
		@info "ckpt-net is of different type: loading weights into larger model..."

		d = net.scales - net′.scales
		for j in 1:net′.scales, w in 1:net′.warps[j]
			net.netarr[d+j][w] = deepcopy(net′[j][w])
		end
		net.netarr[1][end] = deepcopy(net′[1][end])
	end
end

# load data
dstrn = FlyingChairsDataSet(args[:data][:root], augment(;args[:data][:augment]...); split=:trn, args[:data][:params]...)
dsval = FlyingChairsDataSet(args[:data][:root]; split=:val, args[:data][:params]...)
@show dstrn.name, dsval.name

# write args file 
fn = joinpath(args[:train][:savedir], "args.yml")
saveargs(fn, args)

lossfcn = args[:train][:Loss]
if occursin("L1", lossfcn)
	args[:train][:Loss] = L1Loss
elseif occursin("AEE", lossfcn)
	args[:train][:Loss] = AEELoss
else
	@error "Loss function $lossfcn not implemented."
end

train!(net |> device, (trn=dstrn, val=dsval), opt |> device; start=start, device=device, args[:train]...)

# save updated argument file
args[:ckpt] = joinpath(args[:train][:savedir], "net.bson")
saveargs(fn, args)

