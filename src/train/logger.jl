mutable struct Logger
	savedir::String
	best::Dict
end

function Logger(savedir::String, net)
	# init trn.csv, val.csv, backtrack.csv log files
	N = sum(net.warps)
	keys = (:epoch, :lr, :loss, [Symbol("loss$i") for i in 1:N]...)
	values = ([-1], [0], [0], [[0] for i in 1:N]...)
	trnd = (; zip(keys, values)...)
	df = Dict(:trn=>DataFrame(trnd), 
	          :val=>DataFrame(:epoch=>[-1], :lr=>[0], :loss=>[0]))
	for phase ∈ (:trn, :val)
		fn = joinpath(savedir,"$phase.csv")
		safesave(fn, df[phase])
	end
	fn = joinpath(savedir, "backtrack.csv")
	safesave(fn, DataFrame(epoch=[-1], lr=[0], Δρ=[0]))

	# init best train/val losses for backtracking 
	best = Dict(:trn=>Inf, :val=>Inf)
	return Logger(savedir, best)
end

function backtrack!(logger::Logger, epoch::Int, Δ, η)
	ckpt = BSON.load(joinpath(logger.savedir, "net.bson"))
	@assert ckpt[:loss] == logger.best 

	for phase ∈ (:trn, :val)
		fn = joinpath(logger.savedir,"$phase.csv")
		df = DataFrame(load(fn))
		delete!(df, df.epoch .> ckpt[:epoch])
		save(fn, df)
	end

	fn = joinpath(logger.savedir, "backtrack.csv")
	datalog(fn, @sprintf("%d, %.3e, %.3e\n", epoch, Δ, η)) 

	return ckpt[:net], ckpt[:epoch]
end

function (logger::Logger)(phase::Symbol, data::String)
	fn = joinpath(logger.savedir, "$phase.csv")
	datalog(fn, data)
end

function datalog(fn::String, data::String)
	open(fn, "a") do io
		write(io, data)
	end
	return nothing
end
datalog(fn::String, data::Vector) = datalog(fn, join(string.(data),',')*"\n")

