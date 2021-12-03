#=
train.jl
=#

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

function passthrough!(net, data::Dataloader, training=false; β=10f0, opt=missing, desc="", verbose=true, clipnorm=Inf, stopgrad=true, device=identity)
	training && @assert(!ismissing(opt),"Optimizer is required if training.") 
	if training && !isa(clipnorm, Bool) && clipnorm < Inf
		opt = Optimiser(ClipNorm(Float32(clipnorm)), opt)
	end
	P = meter.Progress(length(data), desc=desc, showspeed=true)

	# -- initialize --
	ρ⃗ = zeros(length(data));  # loss history
	Θ = params(net)
	norm∇L = 0

	for (i,F) ∈ enumerate(data)
		F = F |> device
		J = length(F.flows)-1
		if training
			∇L = gradient(Θ) do
				flows = net(F.frame0, F.frame1, J; stopgrad=stopgrad, retflows=true)
				ρ = PiLoss(L1Loss, β, flows, F.flows, F.masks)
			end
			update!(opt, Θ, ∇L)
			norm∇L = verbose ? agradnorm(∇L) : 0
			Π!(net)
		else 
			flow = net(F.frame0, F.frame1, J; retflows=false)
			ρ = EPELoss(flow, F.flows[1], F.masks[1])
		end
		if isnan(ρ) || ρ > 100
			@warn "passthrough!: NaN or large (>100) loss encountered"
			return NaN
		end
		ρ⃗[i] = ρ
		if verbose
			values = [(:loss, @sprintf("%.3e",ρ)), (:avgloss, @sprintf("%.3e",mean(ρ⃗[1:i])))]
			training && push!(values, (:avg_norm∇L, @sprintf("%.3e",norm∇L)))
			training && push!(values, (:test, @sprintf("This should be changing if weights are updating... %.3e",net[2].A[2].weight[20])))
		end
		meter.next!(P; showvalues = verbose ? values : [])
	end
	return ρ⃗
end

# -- Pyramid Iterative Loss (PiLoss) --
function PiLoss(Loss::Function, β::AbstractFloat, flows, flows_gt, masks) 
	J, W = size(flows) .- 1
	loss = 0
	for j=0:J, w=0:W
		loss += 4f0^(-j)*β^(-W+w)*Loss(flows[j+1,w+1], flows_gt[j+1], masks[j+1])
	end
	return loss
end
PiLoss(f, β::Real, args...) = PiLoss(f, Float32(β), args...)
EPELoss(v′,v,M) = mean(√, sum(abs2, M.*(v′.-v), dims=3) .+ 1f-7)
L1Loss(v′,v,M)  = mean(abs, M.*(v′.-v))

function train!(net, loaders, opt; epochs=1, Δval=5, start=1, savedir="./", verbose=true, δ=0.5, γ=0.99, Δsched=1, device=identity, kws...)
	@assert δ < 1 "Backtracking multiplier δ=$δ≮1"

	# best loss
	ρᵇ = Dict(:trn=>Inf, :val=>Inf) 

	# -- initialize files --
	if start == 1 
		fn = joinpath(savedir,"net.bson")
		@info "Saving initial weights to $fn..."
		safesave(fn, Dict(:net=>net, :epoch=>0, :loss=>Dict(:trn=>Inf, :val=>Inf), :η=>opt.eta))
		@info "Creating log files: $savedir/{trn.csv,val.csv,backtrack.csv}"
		init_tracker(savedir)
	end

	# -- training. validate every Δval epochs --
	♪ = start
	while ♪ < start + epochs
		println("--- EPOCH $♪ ---")

		for phase ∈ (:trn, :val)
			if phase == :val && ♪ % Δval != 0
				continue
			end
			desc = "$(string(phase) |> uppercase):"

			# -- main loop --
			ρ⃗ = passthrough!(net, loaders[phase], phase==:trn; opt=opt, desc=desc, verbose=verbose, device=device, kws...)
			ρ̄ = any(isnan.(ρ⃗)) ? NaN : mean(ρ⃗)

			# -- backtracking --
			if isnan(ρ̄) || (phase==:val && δ*ρ̄ > ρᵇ[:val])
				fn = joinpath(savedir,"net.bson")
				if isfile(fn)
					ckpt = load(fn)
					# -- update logs --
					@assert ckpt[:loss][:val] == ρᵇ[:val] "Checkpoint $fn: ckpt[:loss][:val]≠ρᵇ[:val] ($(ckpt[:loss][:val])≠$(ρᵇ[:val]))."
					for phaseᵖ ∈ (:trn, :val)
						if phaseᵖ == :val && ♪ ≤ Δval || phaseᵖ == :train && ♪ ≤ start
							continue
						end
						fn = joinpath(savedir,"$phaseᵖ.csv")
						df = DataFrame(load(fn))
						delete!(df, df.epoch .> ckpt[:epoch])
						save(fn, df)
					end
					log(joinpath(savedir, "backtrack.csv"), @sprintf("%d, %.3e, %.3e\n", ♪, ρ̄-ρᵇ[phase], opt.eta)) 

					# -- rollback train loop --
					net, ♪, ρᵇ = ckpt[:net] |> device, ckpt[:epoch], ckpt[:loss]
					opt.eta *= 0.8 
					@info @sprintf "Backtracking: (η ← %.3e)" opt.eta
					break # phase for-loop
				end
				@error "∄ ckpt $fn for backtracking."
			# -- update best metric --
			elseif ρ̄ < ρᵇ[phase]
				ρᵇ[phase] = ρ̄
				if phase == :val
					@info "Saving best network yet..."
					save(joinpath(savedir,"net.bson"), Dict(:net=>net|>cpu, :epoch=>♪, :loss=>ρᵇ, :η=>opt.eta))
				end
			end

			ρ̄₀ = isnan(ρ̄) ? NaN : mean(ρ⃗)
			logdata = [♪, ρ̄, opt.eta]
			logfmt  = "%d, %.3e, %.2e \n"

			# -- log --
			log(joinpath(savedir, "$phase.csv"), @eval @sprintf($logfmt, $logdata...))
		end

		# -- learning-rate scheduling (η ← γη) --
		if ♪ % Δsched == 0 
			opt.eta *= γ
			@info @sprintf "Scheduling learning rate: (η ← %.3e)" opt.eta
		end
		♪ += 1
	end

	@warn "NOT TESTING. UPDATE BEFORE SUBMITTING JOBS"
	# -- test --
	# if :tst ∈ keys(loaders)
	# 	ρ⃗ = passthrough!(net, loaders[:tst]; desc="TST:", verbose=verbose, kws...)
	# 	ρ̄ = mean(ρ⃗)
	# 	log(joinpath(savedir, "tst.csv"), @sprintf("%s, %.3f\n", loaders[:tst].dataset.name, ρ̄))
	# end

	return nothing
end

function log(fn::String, data::String)
	open(fn, "a") do io
		write(io, data)
	end
	return nothing
end
log(fn::String, data::Vector) = log(fn, join(string.(data),',')*"\n")

function init_tracker(savedir)
	df = Dict(:trn=>DataFrame(epoch=[-1], loss=[0], lr=[0]), :val=>DataFrame(epoch=[-1], loss=[0], lr=[0]))
	for phase ∈ (:trn, :val)
		fn = joinpath(savedir,"$phase.csv")
		safesave(fn, df[phase])
	end
	fn = joinpath(savedir, "backtrack.csv")
	safesave(fn, DataFrame(epoch=[-1], Δρ=[0], lr=[0]))
	return df
end

