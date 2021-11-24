#=
train.jl
=#

function gradnorm(∇::Zygote.Grads)
	gnorm = 0
	for g ∈ ∇
		isnothing(g) && continue
		gnorm += sum(abs2, vec(g))
	end
	return sqrt(gnorm)
end

function passthrough!(net, data::Dataloader, training=false; β=0.5, W=0, opt=missing, desc="", verbose=true, clipnorm=Inf, device=x->x)
	training ? @assert(!ismissing(opt),"Optimizer is required if training.") : nothing
	if training && typeof(clipnorm) <: Real && clipnorm < Inf
		opt = Optimiser(ClipNorm(Float32(clipnorm)), opt)
	end
	P = meter.Progress(length(data), desc=desc, showspeed=true)

	# -- initialize --
	ρ⃗ = zeros(length(data));  # loss history
	Θ = params(net)
	norm∇L = 0

	for (i,F) ∈ enumerate(data)
		F = F |> device
		if training
			∇L, ρ = PyramidGradient(net, Θ, F; β=β, W=W)
			update!(opt, Θ, ∇L)
			norm∇L = verbose ? gradnorm(∇L) : 0
			Π!(net)
		else 
			v = flowctf(net, F.I₀[1], F.I₁[1]; J=length(F.v)-1, W=W)
			ρ = EPELoss(v, F.v[1], F.M[1])
		end
		if isnan(ρ) 
			@warn "passthrough!: NaN loss encountered"
			return NaN
		end
		ρ⃗[i] = ρ
		if verbose
			values = [(:loss, ρ), (:avgloss, mean(ρ⃗[1:i]))]
			training && push!(values, (:norm∇L, norm∇L))
		end
		meter.next!(P; showvalues = verbose ? values : [])
	end
	return ρ⃗
end

function PyramidGradient(net, Θ, F::FlowSample; β=0.5, W=0)
	J = length(F.v)-1
	grads = []
	loss_vec = []

	local WI₁, v, ρ, Δgrad
	v = zero(Float32)
	for j ∈ J:-1:0
		for w ∈ 0:W
			WI₁ = (j==J && w==0) ? F.I₁[j+1] : backward_warp(F.I₁[j+1], v)
			∇L = gradient(Θ) do
				v = net(F.I₀[j+1], WI₁, v)
				ρ = EPELoss(v, F.v[j+1], F.M[j+1])
			end
			pushfirst!(grads, ∇L)
			pushfirst!(loss_vec, ρ)
		end
		v = j>0 ? 2*upsample_bilinear(v, (2,2)) : v
	end

	Δgrad = grads[1]
	loss = loss_vec[1]
	for j=0:J, w=0:W
		j==0 && w==W && continue
		j′= (W+1).*j+W-w+1
		α = 0.25^(W-w)*β^j
		loss += α*loss_vec[j′]
		for p ∈ Θ
			isnothing(grads[j′][p]) && continue
			isnothing(Δgrad[p]) && continue
			Δgrad[p] .+= α*grads[j′][p]
		end
	end
	return Δgrad, loss
end

# -- Pyramid EPE loss --
function PyramidLoss(f::Function,β::Real,v⃗′,v⃗,M) 
	J = length(v⃗)-1
	W = length(v⃗′)÷(J+1)-1
	ρ = 0
	for j=0:J, w=0:W
		j′= (W+1).*j+W-w
		α = 0.25^(W-w)*β^j
		ρ = ρ + α*f(v⃗′[j′+1], v⃗[j+1], M[j+1])
	end
	return ρ
end
EPELoss(v′,v,M) = mean(√, sum(abs2, M.*(v′.-v), dims=3) .+ 1e-7)
PyramidEPELoss(x...) = PyramidLoss(EPELoss,x...)
L1Loss(v′,v,M)  = mean(abs, M.*(v′.-v))
PyramidL1Loss(x...) = PyramidLoss(L1Loss,x...)

function train!(net, loaders, opt; J=0, W=0, β=0.8, epochs=1, Δval=5, start=1, savedir="./", verbose=true, δ=0.5, γ=0.99, Δsched=1, clipnorm=Inf, device=x->x)
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
			ρ⃗ = passthrough!(net, loaders[phase], phase==:trn; β=β, W=W, opt=opt, desc=desc, verbose=verbose, device=device, clipnorm=clipnorm)
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
					opt.eta *= 0.8 #0.8ckpt[:η]
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

	# -- test --
	if :tst ∈ keys(loaders)
		ρ⃗ = passthrough!(net, loaders[:tst]; β=β, W=W, desc="TST:", verbose=verbose, device=device)
		ρ̄ = mean(ρ⃗)
		log(joinpath(savedir, "tst.csv"), @sprintf("%s, %.3f\n", loaders[:tst].dataset.name, ρ̄))
	end

	return net
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

