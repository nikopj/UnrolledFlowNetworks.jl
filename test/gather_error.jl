using NNlib, NNlibCUDA, CUDA, Zygote, Flux
using OMEinsum
using ForwardDiff
using Printf
import UnrolledFlowNetworks as ufn
CUDA.allowscalar(false)

W1 = randn(3,3,1,1)
x1 = randn(3,3,1,1)
idxx = round.(Int, clamp.(4*randn(3,3,1,1), 1, 3))
idxy = round.(Int, clamp.(4*randn(3,3,1,1), 1, 3))
#idx = tuple.(idxx,idxy,1,1)
idx = 2*randn(3,3,2,1)

function net1(W,x,flow)
	@ein img[m,n,i,j] := W[m,k,1,1]*x[k,n,i,j]
	@ein flow[m,n,i,j] := W[m,k,1,1]*flow[k,n,i,j]
	z = ufn.warp_bilinear(img, flow)
	# if x isa CuArray
	# 	flow = ufn.cartesian4(flow1,flow2,zero(flow1).+1,zero(flow2).+1)
	# else
	# 	flow = tuple.(flow1,flow2,1,1)
	# end
	# @show typeof(flow)
	# if eltype(flow) <: NTuple{N, Dual} where N
	# 	@show true
	# 	flow = ForwardDiff.partials.(flow)
	# 	flow = map(x->NTuple{4,Int32}(x), flow)
	# else
	# 	@show false
	# end
	#flow = flow1 + flow2
	#@show flow
	#z = NNlib.gather(img, flow)
	#@ein z[m,n,i,j] := W[m,k,1,1]*z[k,n,i,j]
	return z
end

@printf "CPU\n"
Loss(x) = sum(x)
@show net1(W1,x1,idx)
grad1 = Zygote.gradient(()->Loss(net1(W1,x1,idx)), Params([W1,x1]))
for g ∈ grad1; @show g; end

W2 = W1 |> cu
x2 = x1 |> cu
idx2 = idx |> cu
idxx2 = idxx |> cu
idxy2 = idxy |> cu

@printf "GPU\n"
@show net1(W2,x2,idx2)
grad2 = Zygote.gradient(()->Loss(net1(W2,x2,idx2)), Params([W2,x2]))
for g ∈ grad2; @show g; end

G2 = Array(grad2[W2])
G1 = grad1[W1]
G1 ≈ G2

