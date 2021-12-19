#=
analyze.jl
=#

using UnrolledFlowNetworks, Flux
using Printf
includet("visual.jl")

save_dir = "models/PiBCANet-logthresh_alpha_loss_decay_init-7a"

println("Loading $save_dir...")

curves = training_curves(save_dir)
fig1, _, _ = plot_training_curves(curves)
save(joinpath(save_dir,"training.png"), fig1)

# args = loadargs(joinpath(save_dir, "args.yml"))
ckpt = load(joinpath(save_dir, "net.bson"))
net = ckpt[:net]

mkpath(joinpath(save_dir, "weights"))
for j=0:net.J, w=0:net.W, k=1:net.K
	local A, B, fig2, fig3
	print("\r($j,$w)A[$k]...")
	A = net[j+1,w+1].A[k].weight
	fig2, _, _ = visplot(A, (2, net.M); mosaic=true, resolution=100 .*(net.M + 1, 2), colorbar=true);
	t = join([@sprintf("λ=%.1e", net[j+1,w+1].λ[k][1,1,i,1]) for i ∈ 1:net.M], " "^3)
	Label(fig2[0,:], t, height=0.5)
	fn = joinpath(save_dir,"weights/$(j)_$(w)_A$k.png")
	save(fn, fig2)

	print("\r($j,$w)B[$k]...")
	B = net[j+1,w+1].Bᵀ[k].weight
	fig3, _, _ = visplot(B, (2, net.M); mosaic=true, resolution=100 .*(net.M + 1, 2), colorbar=true);
	Label(fig3[:,0], @sprintf("τ=%.1e", net[j+1,w+1].τ[k][1]), width=0.5, rotation=π/2)
	fn = joinpath(save_dir,"weights/$(j)_$(w)_B$k.png")
	save(fn, fig3)
end


