#==============================================================================
                                LOSSES                             
==============================================================================#

# -- Pyramid Iterative Loss (PiLoss) --
function PiLoss!(lossarr, Loss::Function, α::T, β::T, flows, flows_gt, masks) where {T <: AbstractFloat}
	scales, warps = length(flows), [length(f) for f in flows]
	loss = 0
	k = 1
	for j=1:scales, w=1:warps[j]
		M = ismissing(masks) ? 1 : masks[j+1]
		lossjw = Loss(flows[j][w], flows_gt[j], M)
		loss = loss + α^(j-1)*β^(warps[j]-w)*lossjw
		lossarr[k] = lossjw
		k += 1
	end
	return loss
end
AEELoss(x,y,M) = mean(√, sum(abs2, M.*(x-y), dims=3) .+ 1f-7)
L1Loss(x,y,M)  = mean(abs, M.*(x-y))

# to be defined by others
weight_decay_penalty(net) = 0
project!(net) = nothing
