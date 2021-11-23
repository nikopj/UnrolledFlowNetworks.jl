using Flux, Zygote, Statistics, Test

T = Float32
x = rand(T,3)
W1 = rand(T,3,3)
W2 = rand(T,3,3)

function net1(x)
	z1 = W1*x
	z2 = W2*Zygote.dropgrad(z1)
	return z1, z2
end

function net2(x)
	z1 = W1*x
	z2 = W2*z1
	return z1, z2
end

function net3(x)
	z = []
	z1 = W1*x
	z2 = W2*Zygote.dropgrad(z1)
	#z2 = W2*z1
	Zygote.@ignore push!(z, z1)
	Zygote.@ignore push!(z, z2)
	return z
end

z1, z2 = net(x)
∂L1 = ones(T,3)*x'
∂L11 = ((W2'+I)*ones(T,3))*x'
∂L2 = ones(T,3)*z1'

Loss(z1, z2) = sum(z1.+z2)

grad1 = gradient(Params([W1,W2])) do
	Loss(net1(x)...)
end
grad2 = gradient(Params([W1,W2])) do
	Loss(net2(x)...)
end
grad3 = gradient(Params([W1,W2])) do
	Loss(net3(x)...)
end

@test grad1[W1] ≈ ∂L1 atol=1e-7
@test grad2[W1] ≈ ∂L11 atol=1e-7
@test grad1[W2] ≈ ∂L2 atol=1e-7

