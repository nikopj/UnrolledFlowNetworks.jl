using GLMakie; GLMakie.WINDOW_CONFIG.vsync[] = false
using DataFrames
using FileIO
using Images

function training_curves(save_dir)
	dfload(fn) = begin 
		df = DataFrame(load(joinpath(save_dir, fn)))
		delete!(df, df.epoch .≤ 0)
	end
	dfdict = Dict(:trn=>dfload("trn.csv"), 
	              :val=>dfload("val.csv"), 
	              :backtrack=>dfload("backtrack.csv"))
	return dfdict
end

function plot_training_curves(dfd::Dict)
	# from https://stackoverflow.com/questions/64991944/double-y-axis-on-left-and-right-side-of-plot-in-makie-jl
	fig = Figure(resolution=(1000,600))
	ax1 = GLMakie.Axis(fig[1,1], yaxisposition=:right)
	ax2 = GLMakie.Axis(fig[1,1])
	linkxaxes!(ax1, ax2)
	hidexdecorations!(ax1)
	hidespines!(ax1)

	obj1 = lines!(ax1, dfd[:trn].epoch, dfd[:trn].lr, color=:hotpink, linewidth=5)
	obj2 = scatter!(ax2, dfd[:trn].epoch, dfd[:trn].loss, color=:gray, strokewidth=2)
	obj3 = scatter!(ax2, dfd[:val].epoch, dfd[:val].loss, color=:magenta, strokewidth=2)

	ax1.yticks = range(extrema(dfd[:val].lr)..., length=5) |> x->round.(x, sigdigits=2)

	ax1.ylabel = "learning rate"
	ax2.xlabel = "epoch"
	ax2.ylabel = "loss"

	axislegend(ax1, [obj2, obj3, obj1], ["training PiLoss", "validation EPE", "learning rate"])

	return fig, (ax1, ax2), (obj1, obj2, obj3)
end

function visplot(fplot::Function, fimg::Function, imgs, grid_shape; resolution=missing, titles=missing, colorbar_kws=missing, kws...)
	local fig, plot_objs 
	if !ismissing(resolution)
		fig = Figure(resolution=resolution)
	else
		fig = Figure()
	end
	axes = Matrix{GLMakie.Axis}(undef, grid_shape)
	default_kws = Dict(:interpolate=>false, :colormap=>:grays)

	k = 1
	for i=1:grid_shape[1], j=1:grid_shape[2]
		ax  = GLMakie.Axis(fig[i,j])                                   # get axes at layout (i,j)
		obj = fplot(ax, fimg(imgs[k]); default_kws..., kws...) # plot

		if k==1
			plot_objs = Matrix{typeof(obj)}(undef, grid_shape)
		end
		plot_objs[i,j] = obj

		# set titles
		if !ismissing(titles) && k ≤ length(titles)
			ax.title = titles[k]
		end

		# remove axes border and ticks
		ax.aspect = DataAspect() # lock aspect ratio
		hidespines!(ax)                
		hidedecorations!(ax)

		# store axes and plot objects for return
		axes[i,j] = ax
		plot_objs[i,j] = obj

		k += 1
		k > length(imgs) && break
	end

	# plot colorbar
	if !ismissing(colorbar_kws)
		cb = Colorbar(fig[:,end+1]; colorbar_kws...)
	end

	if length(axes) > 1
		# link panning and zooming between all axes
		linkxaxes!(axes...)
		linkyaxes!(axes...)
	end

	# shows information on hover
	# inspector = DataInspector(fig)

	return fig, axes, plot_objs
end

function visplot(imgs::Array{T,N}, gs=(length(imgs),1); mosaic=false, colorbar=false, kws...) where {T, N}
	# turn batched tensor into vector of tensors, then call visplot
	if N ∈ (3,4) 
		imgvec = [imgs[:,:,Tuple(I)...] for I ∈ CartesianIndices(size(imgs)[3:end])]
		if mosaic
			# put images into single tensor -- don't make subplots
			pad = ceil(Int, size(imgs,1) / 5)
			imgs = mosaicview(imgvec..., nrow=gs[1], ncol=gs[2], npad=pad, rowmajor=true, fillvalue=maximum(imgs))
			imgvec = [imgs]
			gs = (1,1)
		end
		return visplot(imgvec, gs; colorbar=colorbar, kws...)

	# plot images
	elseif T <: Union{Matrix{RGB{N0f8}}, Matrix{Gray{N0f8}}, Matrix{HSV{Float32}}}
		return visplot(image!, rotr90, imgs, gs; kws...)
	end

	limits = minimum(minimum, imgs), maximum(maximum, imgs)

	if colorbar
		colorbar_kws = Dict(:limits=>limits, :colormap=>:grays)
	else
		if kws isa Dict && :colorbar_kws ∈ kws.keys
			colorbar_kws = kws[:colorbar_kws]
			delete!(kws, :colorbar_kws)
		else
			colorbar_kws = missing
		end
	end

	# plot gray-scale values of tensor 
	return visplot(heatmap!, rotr90, imgs, gs; colorrange=limits, colorbar_kws=colorbar_kws, kws...)
end

