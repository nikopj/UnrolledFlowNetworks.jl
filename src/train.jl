#=
train.jl
=#

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

function passthrough!(net, data::Dataloader, training=false; Loss=L1Loss, gt_init=false, weight_decay=0, α=4f0, β=10f0, opt=missing, desc="", verbose=true, clipnorm=Inf, clipvalue=Inf, stopgrad=true, device=identity, use_mask=true, maxiter=Inf)
	clipnorm  = (clipnorm == false)  ? Inf32 : Float32(clipnorm)
	clipvalue = (clipvalue == false) ? Inf32 : Float32(clipvalue)
	weight_decay = Float32(weight_decay)
	α = Float32(α)
	β = Float32(β)

	training && @assert(!ismissing(opt),"Optimizer is required if training.") 
	if training && clipvalue < Inf
		opt = Optimiser(ClipValue(Float32(clipvalue)), opt)
	end
	N = training ? min(maxiter, length(data)) : length(data)
	P = meter.Progress(N, desc=desc, showspeed=true)

	# -- initialize --
	ρ⃗ = zeros(N);  # loss history
	Θ = params(net)
	total_gnorm = 0

	if weight_decay == 0
		penalty = ()->0
	else
		penalty = ()-> begin 
			local loss
			loss = zero(Float32)
			for j ∈ 0:net.J
				for w ∈ 0:net.W
					for k ∈ 1:net.K
						loss += sum(abs2, net[j+1,w+1].A[k].weight) + sum(abs2, net[j+1,w+1].Bᵀ[k].weight)
					end
					net.shared_iter && break
				end
				net.shared_scale && break
			end
			return loss
		end
	end

	for (i,F) ∈ enumerate(data)
		local wdpen
		J = length(F.flows)-1
		if training
			∇L = gradient(Θ) do
				flows = net(F.frame0, F.frame1, J; v̄= gt_init ? F.flows[J+1] : missing, stopgrad=stopgrad, retflows=true)
				wdpen = penalty()
				ρ = PiLoss(Loss, α, β, flows, F.flows, use_mask ? F.masks : missing) + weight_decay*wdpen 
			end
			clip_total_gradnorm!(∇L, clipnorm)
			total_gnorm = verbose ? gradnorm(∇L) : 0
			update!(opt, Θ, ∇L)
			Π!(net)
		else 
			flow = net(F.frame0, F.frame1, J; v̄= gt_init ? F.flows[J+1] : missing, retflows=false)
			ρ = EPELoss(flow, F.flows[1], use_mask ? F.masks[1] : 1)
		end

		if isnan(ρ) #|| ρ > 1e3
			@warn "passthrough!: NaN or large (>100) loss encountered"
			return NaN
		end

		ρ⃗[i] = ρ
		if verbose
			values = [(:loss, @sprintf("%.3e",ρ)), (:avgloss, @sprintf("%.3e",mean(ρ⃗[1:i])))]
			training && push!(values, (:total_gnorm, @sprintf("%.3e",total_gnorm)))
			training && push!(values, (:weight_decay, @sprintf("%.3e", wdpen)))
			push!(values, (:test, @sprintf("This should be changing if weights are updating... %.3e",sum(net[1].A[end].weight))))
		end

		meter.next!(P; showvalues = verbose ? values : [])
		if training && i ≥ maxiter
			shuffle!(data)
			break
		end

		CUDA.unsafe_free!(F.frame0)
		CUDA.unsafe_free!(F.frame1)
		CUDA.unsafe_free!.(F.flows)
		CUDA.unsafe_free!.(F.masks)
	end
	return ρ⃗
end

# -- Pyramid Iterative Loss (PiLoss) --
function PiLoss(Loss::Function, α::T, β::T, flows, flows_gt, masks) where {T <: AbstractFloat}
	J, W = size(flows) .- 1
	loss = 0
	for j=0:J, w=0:W
		M = ismissing(masks) ? 1 : masks[j+1]
		loss = loss + α^(-j)*β^(-W+w)*Loss(flows[j+1,w+1], flows_gt[j+1], M)
	end
	return loss 
end
PiLoss(f, α::Real, β::Real, args...) = PiLoss(f, Float32(α), Float32(β), args...)
EPELoss(x,y,M) = mean(√, sum(abs2, M.*(x-y), dims=3) .+ 1f-7)
L1Loss(x,y,M)  = mean(abs, M.*(x-y))

function train!(net, loaders, opt; epochs=1, Δval=5, start=1, savedir="./", verbose=true, δ=0.5, γ=0.99, Δsched=1, device=identity, kws...)
	@assert δ > 1 "Backtracking multiplier δ=$δ must be greater than 1."
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
			if isnan(ρ̄) || (phase==:val && ρ̄ > δ*ρᵇ[:val])
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
					datalog(joinpath(savedir, "backtrack.csv"), @sprintf("%d, %.3e, %.3e\n", ♪, ρ̄-ρᵇ[phase], opt.eta)) 

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
			datalog(joinpath(savedir, "$phase.csv"), @eval @sprintf($logfmt, $logdata...))
		end

		# -- learning-rate scheduling (η ← γη) --
		if ♪ % Δsched == 0 
			opt.eta *= γ
			@info @sprintf "Scheduling learning rate: (η ← %.3e)" opt.eta
		end
		♪ += 1
	end

	# -- test --
	if :tst ∈ keys(loaders)
		ρ⃗ = passthrough!(net, loaders[:tst]; desc="TST:", verbose=verbose, device=device, kws...)
		ρ̄ = mean(ρ⃗)
		datalog(joinpath(savedir, "tst.csv"), @sprintf("%s, %.3f\n", loaders[:tst].dataset.name, ρ̄))
	end

	return nothing
end

function datalog(fn::String, data::String)
	open(fn, "a") do io
		write(io, data)
	end
	return nothing
end
datalog(fn::String, data::Vector) = datalog(fn, join(string.(data),',')*"\n")

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

