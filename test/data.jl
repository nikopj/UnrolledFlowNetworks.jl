using UnrolledFlowNetworks
using ImageView, Plots

ds = MPISintelDataset("/home/nikopj/dataset/MPI_Sintel/", split="val", gray=true)

J = 2
scale = 1
σ = 1
dltrn = Dataloader(ds, true; batch_size=3, crop_size=256, scale=scale, J=J, σ=σ)
dlval = Dataloader(ds, false; batch_size=1, scale=scale, J=J, σ=σ)

F = dltrn[rand(1:length(dltrn))]
WI₁ = [backward_warp(F.I₁[j], F.v[j]) for j ∈ 1:J+1]
frame_vec = [cat(tensor2img.((F.I₀[j], F.M[j].*WI₁[j], F.I₁[j]))..., dims=4) for j ∈ 1:J+1];
flow_vec = colorflow.(F.v)





