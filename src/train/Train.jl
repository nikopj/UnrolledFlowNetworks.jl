module Train

using Flux
using Flux.Optimise: Optimiser, ClipValue, update!
using DataLoaders: DataLoader, nobs, getobs, shuffleobs
using CUDA: CuIterator
using Zygote

using Statistics
using Printf
using FileIO
import BSON
using CSV, CSVFiles
using DataFrames
using DrWatson: safesave
import ProgressMeter as Meter

include("../utils/Utils.jl")
using .Utils

include("loops.jl")
export train!, passthrough!

include("losses.jl")
export PiLoss, AEELoss, L1Loss

include("logger.jl")
include("utils.jl")

end 
