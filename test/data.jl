using UnrolledFlowNetworks, Test, Statistics
import UnrolledFlowNetworks as ufn
#using ImageView, Plots

#ds = MPISintelDataset("dataset/MPI_Sintel/"; split="val", gray=true)

J = 3
σ = 0
# dl_trn = Dataloader(ds, true; batch_size=3, crop_size=256, scale=scale, J=J, σ=σ)
# dl_val = Dataloader(ds, false; batch_size=1, scale=scale, J=J, σ=0)

dl = Dataloader(ds_val, true; batch_size=6, crop_size=256, scale=0, J=J, σ=σ, device=device)

F = dl[rand(1:length(dl))]
@show F
J = length(F.flows)-1

H = ConvGaussian(1; stride=2) |> device
frames0 = [F.frame0]
frames1 = [F.frame1]
for j=1:J
	push!(frames0, H(frames0[j]))
	push!(frames1, H(frames1[j]))
end

Wu1 = [warp_bilinear(frames1[j], F.flows[j]) for j ∈ 1:J+1]

for j ∈ 1:J+1
	@show size(frames0[j])
	@show (j-1, mean(abs, F.masks[j].*(Wu1[j] - frames0[j])) )
end

# frame_vec = [cat(tensor2img.((F.I₀[j], F.M[j].*WI₁[j], F.I₁[j]))..., dims=4) for j ∈ 1:J+1];
# flow_vec = colorflow.(F.v)





