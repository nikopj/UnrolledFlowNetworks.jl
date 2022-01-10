#=
analyze.jl
=#

using UnrolledFlowNetworks, Flux, FileIO
using DrWatson
import UnrolledFlowNetworks as ufn
using Printf
include("visual.jl")

fn = isinteractive() ? "models/PiBCANet-scale=5_J=0_flomax_cs_K-3b/args.yml" : ARGS[1]
args = loadargs(fn)
save_dir = dirname(fn)

println("Loading $fn...")
ckpt = load(joinpath(save_dir, "net.bson"))
net = ckpt[:net]
@show net

train_plot = true
weights = true
thresholds = true
passthrough = true

if train_plot
	fn = joinpath(save_dir, "training.png")
	println("Saving training curves to $fn...")
	curves = training_curves(save_dir)
	plot_training_curves(curves, Figure(resolution=(1000,600)))
	save(fn, current_figure())
end

if weights
	fn = joinpath(save_dir, "A.png")
	println("Saving A weights to $fn...")
	A = [begin
		am = batch_mosaicview(cat([net[j,w].A[k].weight for k=1:net.K]..., dims=4), nrow=net.M, ncol=net.K, npad=1)
		colorflow(am[:,:,:,:])
	end for j=1:net.J+1, w=1:net.W+1]
	visplot(A...; grid_shape=(net.J+1, net.W+1))
	save(fn, current_figure())

	fn = joinpath(save_dir, "B.png")
	println("Saving Bᵀ weights to $fn...")
	B = [begin
		bm = batch_mosaicview(cat([net[j,w].Bᵀ[k].weight for k=1:net.K]..., dims=4), nrow=net.M, ncol=net.K, npad=1)
		colorflow(bm[:,:,:,:])
	end for j=1:net.J+1, w=1:net.W+1]
	visplot(B...; grid_shape=(net.J+1, net.W+1))
	save(fn, current_figure())
end

if thresholds
	fn = joinpath(save_dir, "thresh.png")
	println("Saving thresholds λ to $fn...")
	λv = [cat(net[j,w].λ..., dims=4)[1,1,:,:] .|> exp for j=1:net.J+1, w=1:net.W+1] 
	clamp!.(λv, 0, 1) 
	visplot(λv...; grid_shape=(net.J+1, net.W+1), colorbar=true)
	save(fn, current_figure())

	fn = joinpath(save_dir, "step.png")
	println("Saving step-sizes τ to $fn...")
	τv = [cat(net[j,w].τ..., dims=4)[1,1,:,:] .|> exp for j=1:net.J+1, w=1:net.W+1]
	clamp!.(τv, 0, 1)
	visplot(τv...; grid_shape=(net.J+1, net.W+1), colorbar=true)
	save(fn, current_figure())
end

function PT(u0, u1, vgt; name="", save_dir=".")
	println("Running $name passthrough...")
	vgt_c = colorflow(vgt)
	maxflow = maximum(sqrt.(sum(abs2, vgt, dims=3)))

	vhat = net(u0, u1, net.J; retflows=false) |> collect
	loss = EPELoss(vhat, vgt, 1)

	u = (u0 + u1)./2f0 
	vhat_c = colorflow(vhat, maxflow=maxflow)
	titles = ["sum", @sprintf("EPE=%.3f",loss), "flow gt"]

	fn = joinpath(save_dir, "passthrough", "$(name)_compare.png")
	visplot(tensor2img(u), vhat_c, vgt_c; titles=titles)
	safesave(fn, current_figure())

	println("Running $name passthrough with retflows...")
	vgtp = begin
		pad = ufn.calcpad(size(vgt)[1:2], 2^net.J)
		pad_reflect(vgt, pad, dims=(1,2))
	end
	netflows = net(u0, u1, net.J; retflows=true)
	netflows_c = colorflow.([netflows[j+1,w+1] * 2^j for j=0:net.J, w=0:net.W], maxflow=maxflow)

	uP = begin
		pad = ufn.calcpad(size(u)[1:2], 2^net.J)
		pad_reflect(u, pad, dims=(1,2))
	end
	uP0 = begin
		pad = ufn.calcpad(size(u0)[1:2], 2^net.J)
		pad_reflect(u0, pad, dims=(1,2))
	end
	uP1 = begin
		pad = ufn.calcpad(size(u1)[1:2], 2^net.J)
		pad_reflect(u1, pad, dims=(1,2))
	end
	pysum = get_pyramid(uP, net.J) .|> tensor2img
	pyu0 = get_pyramid(uP0, net.J) .|> tensor2img
	pyu1 = get_pyramid(uP1, net.J) .|> tensor2img
	pyflowgt = get_pyramid(vgtp, net.J) ./ [2^j for j=0:net.J]
	pyflowgt_c = colorflow.(pyflowgt .* [2^j for j=0:net.J], maxflow=maxflow)

	titles = [@sprintf("EPE=%.3f",EPELoss(netflows[j,w], pyflowgt[j], 1)) for j=1:net.J+1, w=1:net.W+1]
	#visplot(pysum...; link=false)
	visplot(pyu0...; link=false)
	F = current_figure()
	visplot(pyu1...; fig=F[:,end+1], link=false)
	visplot(netflows_c...; fig=F[:,end+1], titles=titles, grid_shape=(net.J+1, net.W+1), link=false)
	visplot(pyflowgt_c...; link=false, fig=F[:,end+1])
	fn = joinpath(save_dir, "passthrough", "$(name)_flows_compare.png")
	safesave(fn, current_figure())
	return netflows
end

if passthrough
	mkpath(joinpath(save_dir, "passthrough"))
	u0 = tensorload("dataset/MPI_Sintel/training/clean/shaman_2/frame_0048.png", gray=true)
	u1 = tensorload("dataset/MPI_Sintel/training/clean/shaman_2/frame_0049.png", gray=true)
	vgt= tensorload("dataset/MPI_Sintel/training/flow/shaman_2/frame_0048.flo")
	scale = args[:data][:dl_params][:scale]
	u0 = get_pyramid(u0, scale)[end]
	u1 = get_pyramid(u1, scale)[end]
	vgt= get_pyramid(vgt,scale)[end] / 2^scale
	PT(u0,u1,vgt; name="shaman248_scale=$scale", save_dir=save_dir)

	ds = FlyingChairsDataset("dataset/FlyingChairs"; split="val", args[:data][:ds_params]...)
	dl = Dataloader(ds, false; batch_size=1, J=net.J, scale=args[:data][:dl_params][:scale])
	F = dl[rand(1:length(dl))]
	PT(F.frame0, F.frame1, F.flows[1]; name="Fval", save_dir=save_dir)

	M, N, _, _ = size(F.frame0)
	b1 = M÷4
	b2 = N÷3
	d  = N÷6
	@show M, N, b1, b2, d
	u0 = zeros(Float32, M,N,1,1)
	u1 = copy(u0)
	u0[M÷3:M÷3+b1, N÷3:N÷3+b2] .= 1
	u1[M÷3:M÷3+b1, N÷3+d:N÷3+b2+d] .= 1
	vgt = zeros(Float32, M,N,2,1)
	vgt[u0[:,:,1,1] .== 1, 2, 1] .= d
	PT(u0, u1, vgt; name="box", save_dir=save_dir)
end

