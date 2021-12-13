#=
analyze.jl
=#

using UnrolledFlowNetworks, Flux
using Printf
includet("visual.jl")

save_dir = "models/PiBCANet-shared_lr_t0_W=5-12a"

curves = training_curves(save_dir)
fig1, _, _ = plot_training_curves(curves)
save(joinpath(save_dir,"training.png"), fig1)


# args = loadargs(joinpath(save_dir, "args.yml"))
ckpt = load(joinpath(save_dir, "net.bson"))
net = ckpt[:net]

# mkpath(joinpath(save_dir, "weights"))
# for k ∈ 1:net.K
# 	local A, B, fig2, fig3
# 	print("\rA[$k]...")
# 	A = net[1].A[k].weight
# 	fig2, _, _ = visplot(A, (2, net.M); resolution=100 .*(net.M + 1, 2), colorbar=true);
# 	t = join([@sprintf("λ=%.1e", net[1].λ[k][1,1,i,1]) for i ∈ 1:net.M], " "^3)
# 	Label(fig2[0,:], t, height=0.5)
# 	fn = joinpath(save_dir,"weights/A$k.png")
# 	save(fn, fig2)
# 
# 	print("\rB[$k]...")
# 	B = net[1].Bᵀ[k].weight
# 	fig3, _, _ = visplot(B, (2, net.M); resolution=100 .*(net.M + 1, 2), colorbar=true);
# 	Label(fig3[:,0], @sprintf("τ=%.1e", net[1].τ[k][1]), width=0.5, rotation=π/2)
# 	fn = joinpath(save_dir,"weights/B$k.png")
# 	save(fn, fig3)
# end


