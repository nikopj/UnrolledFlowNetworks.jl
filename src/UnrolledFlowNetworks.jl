module UnrolledFlowNetworks

using Flux
using CUDA
using NNlib
using Statistics
using Zygote

include("utils/Utils.jl")
using .Utils
export loadargs, saveargs, awgn
export warp_bilinear, pyramid, ConvGaussian, ConvGradient

# include("visual.jl")
# using .Visual
# export visplot

include("data/Data.jl")
using .Data
export tensor2img, img2tensor, tensorload
export FlyingChairsDataSet, augment

include("train/Train.jl")
using .Train
export passthrough!, train!, AEELoss, L1Loss

include("solvers.jl")
using .Solvers
export flow_ictf, powermethod

include("networks.jl")
export PiBCANet

end
