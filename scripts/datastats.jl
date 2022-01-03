using UnrolledFlowNetworks
using ProgressMeter
using Statistics
using FileIO

vfn = begin
	flt = x->occursin(".flo", x)
	fns = readdir("dataset/FlyingChairs/data/", join=true)
	filter(flt, fns)
end

df = DataFrame(min=[], med=[], max=[])

@showprogress for fn ∈ vfn
	local flow, mg
	flow = tensorload(fn)
	mg = sqrt.(sum(abs2, flow, dims=3))
	push!(df, [minimum(mg), median(mg), maximum(mg)])
end

save("stats.csv", df)
