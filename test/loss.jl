using NNlib, NNlibCUDA, CUDA, Zygote, Flux
using OMEinsum
using Printf
using UnrolledFlowNetworks
import UnrolledFlowNetworks as ufn
CUDA.allowscalar(false)

#device = CUDA.functional() ? begin @info "Using GPU."; gpu end : cpu
#device = begin @info "Using CPU"; cpu end

# gray = true
# root = "dataset/MPI_Sintel/training/clean/shaman_2/"
# T = Float32
# u₀ = tensorload(T , joinpath(root,"frame_0048.png"); gray=gray) |> device
# u₁ = tensorload(T , joinpath(root,"frame_0049.png"); gray=gray) |> device
# vgt= tensorload(T , "dataset/MPI_Sintel/training/flow/shaman_2/frame_0048.flo") |> device

ds = MPISintelDataset("dataset/MPI_Sintel/", split="val", gray=true)
dl1 = Dataloader(ds, false; batch_size=1, crop_size=400, scale=0, J=3, σ=0, device=cpu)
dl2 = Dataloader(ds, false; batch_size=1, crop_size=400, scale=0, J=3, σ=0, device=gpu)

dl2.minibatches = dl1.minibatches

dl = [dl1, dl2]

@show dl1[1].frame0 ≈ Array(dl2[1].frame0)
@show dl1[1].frame1 ≈ Array(dl2[1].frame1)
@show ufn.flip(dl1[1].frame0, 1) ≈ Array( ufn.flip(dl2[1].frame0, 1) )

net = PiBCANet(; W=1, shared=false, K=5, M=4, P=7, s=1, init=true) 
@show net

for (i, device) in enumerate([cpu, gpu])
	global u0, u1, vgt, mask, F, flows
	global net, G, Θ
	i == 1 && @info "CPU"
	i == 2 && @info "GPU"

	F = dl[i][1]
	u0 = F.frame0    
	u1 = F.frame1    
	vgt= F.flows
	mask= F.masks

	net = net |> device
	J = length(vgt) - 1

	Θ = Flux.params(net)
	Loss(flows) = ufn.PiLoss(ufn.EPELoss, 4f0, 10f0, flows, vgt, mask)
	flows = net(u0, u1, J; stopgrad=true, retflows=true)
	@show Loss(flows)

	G = Zygote.gradient(()->Loss(net(u0, u1, 0; stopgrad=true, retflows=true)), Θ)
	
	@show ufn.agradnorm(G);
end
