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

function plot_training_curves(dfd::Dict, F=Figure())
	# from https://stackoverflow.com/questions/64991944/double-y-axis-on-left-and-right-side-of-plot-in-makie-jl
	ax1 = GLMakie.Axis(F[1,1], yaxisposition=:right)
	ax2 = GLMakie.Axis(F[1,1])
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

	return F, (ax1, ax2), (obj1, obj2, obj3)
end

function visplot(fplot::Function, fimg::Function, imgs...; fig=Figure(), grid_shape=(length(imgs),1), titles=missing, colorbar_kws=missing, link=true, kws...)
	local plot_objs 
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
		cb = Colorbar(fig[:,grid_shape[2]+1]; colorbar_kws...)
	else
		cb = missing
	end

	if length(axes) > 1 && link
		# link panning and zooming between all axes
		linkxaxes!(axes...)
		linkyaxes!(axes...)
	end

	# shows information on hover
	# inspector = DataInspector(fig)

	return fig, axes, plot_objs, cb
end

function visplot(imgs...; grid_shape=(length(imgs),1), mosaic=false, colorbar=false, kws...) 
	T = eltype(imgs[1])
	N = ndims(imgs[1])
	# turn batched tensor into vector of tensors, then call visplot
	if length(imgs)==1 && N ∈ (3,4) 
		imgvec = [imgs[1][:,:,Tuple(I)...] for I ∈ CartesianIndices(size(imgs)[3:end])]
		if mosaic
			# put images into single tensor -- don't make subplots
			pad = ceil(Int, size(imgs,1) / 5)
			imgs = mosaicview(imgvec...; nrow=grid_shape[1], ncol=grid_shape[2], npad=pad, rowmajor=true, fillvalue=maxval)
			imgvec = [imgs]
			grid_shape = (1,1)
		end
		return visplot(imgvec...; grid_shape=grid_shape, colorbar=colorbar, kws...)
	# plot images
	elseif T <: Union{RGB{N0f8}, Gray{N0f8}, HSV{Float32}}
		return visplot(image!, rotr90, imgs...; grid_shape=grid_shape, kws...)
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
	return visplot(heatmap!, rotr90, imgs...; grid_shape=grid_shape, colorrange=limits, colorbar_kws=colorbar_kws, kws...)
end

function batch_mosaicview(A::Array{T,4}; kws...) where T
	cat([mosaicview(selectdim(A,3,i); kws...) for i=1:size(A,3)]..., dims=3)
end

