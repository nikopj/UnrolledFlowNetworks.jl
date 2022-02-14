
function passthrough!(net, data, training=false; 
    Loss=L1Loss, 
    gtinit=false, 
    γ=0,
    α=4f0, 
    β=10f0, 
    σ=0,
    opt=missing, 
    desc="", 
    verbose=true, 
    clipnorm=Inf, 
    clipvalue=Inf, 
    stopgrad=true, 
    usemask=true, 
    batchsize=1,
    resolution=1,
    maxiter=Inf,
    device=identity,
    freeze=false) 

	dataloader = DataLoader(data, batchsize; partial=false)

	clipnorm  = (clipnorm == false)  ? Inf32 : Float32(clipnorm)
	clipvalue = (clipvalue == false) ? Inf32 : Float32(clipvalue)

	γ = Float32(γ) # weight decay 
	α = Float32(α) # scale PiLoss penalty
	β = Float32(β) # warp PiLoss penalty
	σ = Float32(σ) # noise-level

	if training && clipvalue < Inf
		opt = Optimiser(ClipValue(Float32(clipvalue)), opt)
	end

	niter = training ? min(maxiter, length(dataloader)) : length(dataloader)
	if verbose
		meter = Meter.Progress(niter, desc=desc, showspeed=true)
	end

	# -- initialize --
	ρ⃗ = zeros(niter);  # loss history

	# train only BCANet at finest scale, latest warp
	if freeze
		Θ = params(net[1,net.warps[1]])
		penalty = γ==0 ? ()->0 : ()->weight_decay_penalty(net[1,net.warps[1]])
	# train everything
	else
		Θ = params(net)    # network parameters
		penalty = γ==0 ? ()->0 : ()->weight_decay_penalty(net)
	end

	# loss fcn penalty

	local ρ, wdpen, ρarr, total_gradnorm
	ρ⃗arr = training ? zeros(sum(net.warps), niter) : nothing

	# main loop
	for (i, batch) ∈ enumerate(dataloader)
		img1, img2, flow, mask = batch
		img1, img2, flow, mask = (img1, img2, flow, mask) .|> device


		img1 = pyramid(awgn(img1, σ)[1], resolution, net.blurimg)[end]
		img2 = pyramid(awgn(img2, σ)[1], resolution, net.blurimg)[end]

		pyflow = pyramid(flow, net.scales + resolution - 1, net.blurflo)[resolution:end] ./ 2^(net.scales + resolution - 2)
		pymask = usemask ? pyramid(mask, net.scales + resolution - 1, net.blurimg)[resolution:end] : missing
		v̄ = gtinit ? pyflow[end] : missing

		# train or val
		if training
			∇L = gradient(Θ) do
				flowarr = net(img1, img2, v̄; stopgrad=stopgrad, retflows=true)
				wdpen = penalty()
				ρarr = Zygote.Buffer(Vector{Float32}(undef, sum(net.warps)))
				ρ = PiLoss!(ρarr, Loss, α, β, flowarr, pyflow, pymask) + γ*wdpen
			end
			# copy stats 
			ρ⃗arr[:,i] = copy(ρarr)

			# clip gradients
			clip_total_gradnorm!(∇L, clipnorm)
			total_gradnorm = verbose ? gradnorm(∇L) : 0

			# projected gradient descent
			update!(opt, Θ, ∇L)
			project!(net)
		else 
			flowhat = net(img1, img2, v̄; retflows=false)
			ρ = AEELoss(flowhat, pyflow[1], usemask ? mask : 1f0)
		end
		ρ⃗[i] = ρ

		if isnan(ρ)
			@warn "passthrough!: NaN encountered"
			return NaN
		end

		if verbose
			values = [(:loss, @sprintf("%.3e",ρ)), (:avgloss, @sprintf("%.3e",mean(ρ⃗[1:i])))]
			if training
				push!(values, (:total_gnorm, @sprintf("%.3e",total_gradnorm)))
				push!(values, (:weight_decay, @sprintf("%.3e", wdpen)))
				for k in 1:sum(net.warps)
					push!(values, (Symbol("loss$k"), @sprintf(" %.2e ", mean(ρ⃗arr[k,1:i]))))
				end
				push!(values, (:test, @sprintf("πnet[end,1] weights updating...? %.3e", sum(net.netarr[end][1].A[end].weight))))
				push!(values, (:test, @sprintf("πnet[1,end] weights updating...? %.3e", sum(net.netarr[1][end].A[end].weight))))
			end
			Meter.next!(meter; showvalues = verbose ? values : [])
		end

		if training && i ≥ maxiter
			break
		end
	end
	return ρ⃗, ρ⃗arr
end

function train!(net, datasets, opt; 
                epochs=1, 
                Δval=5, 
                start=1, 
                savedir="./", 
                verbose=true, 
                δbacktrack=0.5, 
                δsched=0.95, 
                Δsched=1, 
                device=identity, 
                kws...)
	# -- initialize files --
	if start == 1 
		fn = joinpath(savedir,"net.bson")
		@info "Saving initial weights to $fn..."
		safesave(fn, Dict(:net=>net, :epoch=>0, :loss=>Dict(:trn=>Inf, :val=>Inf), :η=>opt.eta))
		@info "Creating log files: $savedir/{trn.csv,val.csv,backtrack.csv}"
		logger = Logger(savedir, net)
	else
		best = Dict(:trn=>Inf, :val=>Inf)
		for phase ∈ (:trn, :val)
			df = joinpath(savedir, "$phase.csv") |> load |> DataFrame
			best[phase] = minimum(df[phase].loss)
		end
		logger = Logger(savedir, best)
	end

	# -- training. validate every Δval epochs --
	epoch = start
	while epoch < start + epochs
		println("--- EPOCH $epoch ---")

		for phase ∈ (:trn, :val)
			if phase == :val && epoch % Δval != 0
				continue
			end
			desc = "$(string(phase) |> uppercase):"

			# -- main loop --
			data = phase==:trn ? shuffleobs(datasets[:trn]) : datasets[:val]
			ρ⃗, ρ⃗arr = passthrough!(net, data, phase==:trn; opt=opt, desc=desc, verbose=verbose, device=device, kws...)

			ρ̄ = any(isnan.(ρ⃗)) ? NaN : mean(ρ⃗)
			if any(isnan.(ρ⃗)) || phase==:val 
				ρ̄arr = NaN
			else
				ρ̄arr = mean(ρ⃗arr, dims=2)
			end

			# -- backtracking --
			if isnan(ρ̄) || (phase==:val && ρ̄ > δbacktrack * logger.best[:val])
				net, epoch = backtrack(logger, ρ̄, opt.eta)
				net = net |> device
				opt.eta *= 0.8 
				@info @sprintf "Backtracking: (η ← %.3e)" opt.eta
				break # phase for-loop

			# -- update best metric --
			elseif ρ̄ ≤ logger.best[phase]
				logger.best[phase] = ρ̄
				if phase == :val
					@info "Saving best network yet..."
					save(joinpath(savedir,"net.bson"), Dict(:net=>net|>cpu, :epoch=>epoch, :loss=>logger.best[:val], :η=>opt.eta))
				end
			end

			if phase == :trn
				logdata = [epoch, opt.eta, ρ̄, ρ̄arr...]
				logfmt  = "%d, %.2e, %.3e"*" %.3e"^length(ρ̄arr) *" \n"
			elseif phase == :val
				logdata = [epoch, opt.eta, ρ̄]
				logfmt  = "%d, %.2e, %.3e \n"
			end
			logger(phase, @eval @sprintf($logfmt, $logdata...))
		end

		# -- learning-rate scheduling (η ← γη) --
		if epoch % Δsched == 0 
			opt.eta *= δsched
			@info @sprintf "Scheduling learning rate: (η ← %.3e)" opt.eta
		end

		epoch += 1
	end

	# -- test --
	if :tst ∈ keys(datasets)
		ρ⃗, _ = passthrough!(net, datasets[:tst]; desc="TST:", verbose=verbose, kws...)
		ρ̄ = mean(ρ⃗)
		logger(:tst, @sprintf("%s, %.3f\n", datasets[:tst].data.data.name, ρ̄))
	end

	return nothing
end
