#==============================================================================
                                UTILS                             
==============================================================================#

# average gradient norm
function agradnorm(∇::Zygote.Grads)
	agnorm = 0
	i = 0
	for g ∈ ∇
		isnothing(g) && continue
		agnorm += norm(vec(g))
		i += 1
	end
	return agnorm / i
end

# total gradient norm (of vectorized params)
function gradnorm(∇::Zygote.Grads)
	gnorm = 0
	for g ∈ ∇
		isnothing(g) && continue
		gnorm += sum(abs2, vec(g))
	end
	return sqrt(gnorm)
end

function clip_total_gradnorm!(∇::Zygote.Grads, thresh)
	gnorm = gradnorm(∇)
	if gnorm > thresh
		for g ∈ ∇
			isnothing(g) && continue
			g .*= thresh/gnorm
		end
	end
	return gnorm
end

