using Flux, Zygote, Statistics
using CUDA
using UnrolledFlowNetworks
import UnrolledFlowNetworks as ufn
using OMEinsum

CUDA.allowscalar(false)

device = gpu
x = randn(Float32, 2,2,3,2) |> device
y = randn(Float32, 2,2,3,2) |> device
v = 5*randn(Float32, 2,2,2,2) |> device
W1 = randn(Float32, 2,2,1,1) |> device
W2 = randn(Float32, 2,2,1,1) |> device
W3 = randn(Float32, 2,2,1,1) |> device

function net(x,v) 
	#@ein v[m,n,i,j] := W1[m,k,1,1]*v[k,n,i,j]
	#@ein z[m,n,i,j] := W2[m,k,1,1]*x[k,n,i,j]
	z = x
	z = ufn.warp_bilinear(z, v)
	@ein z[m,n,i,j] := W3[m,k,1,1]*z[k,n,i,j]
	return z
end

Loss(x,y) = sum(x-y)

grad = gradient(Params([W1,W2,W3])) do
	Loss(net(x,v), y)
end

for g ∈ grad; @show g; end
