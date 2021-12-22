using Flux, UnrolledFlowNetworks, FileIO, CUDA
using Printf
import ProgressMeter as meter

device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu

λ = 2e-1
J = 5
kws = Dict(:retflows=>false, :γ=>100, :β=>Inf, :verbose=>false, :maxit=>20, :maxwarp=>5, :tol=>1e-3, :tolwarp=>1e-5)

ds = MPISintelDataset("dataset/MPI_Sintel"; type="final", split="all", gray=true)
dl = Dataloader(ds, false; batch_size=1, device=device)

@show dl

P = meter.Progress(length(dl), showspeed=true)

loss = 0
for (i, F) ∈ enumerate(dl)
	global loss, P
	local vhat, u0p, u1p, params
	u0p, u1p, params = UnrolledFlowNetworks.preprocess(F.frame0, F.frame1, 2^J)
	vhat = flow_ictf(u0p, u1p, λ, J; kws...)
	vhat = UnrolledFlowNetworks.unpad(vhat, params[2]) 
	loss += EPELoss(vhat , F.flows[1] , F.masks[1])

	meter.next!(P; showvalues=[(:loss, @sprintf("%.3e", loss/i))])
end

