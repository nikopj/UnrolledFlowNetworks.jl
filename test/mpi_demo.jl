using OpticalFlowUtils
import UnrolledFlowNetworks as ufn
using FileIO, Plots
using Flux

root = "/home/nikopj/dataset/MPI_Sintel/training/"

img0  = ufn.tensorload(joinpath(root, "clean/ambush_2/frame_0001.png"), gray=true)
img1  = ufn.tensorload(joinpath(root, "clean/ambush_2/frame_0002.png"), gray=true)
flow = joinpath(root, "flow/ambush_2/frame_0001.flo") |> load
flow = permutedims(flow, (2,3,1)) |> Flux.unsqueeze(3)
occlusion = ufn.tensorload(joinpath(root, "occlusions/ambush_2/frame_0001.png"))

Wimg = ufn.backward_warp(img1, flow)

x = cat(img0, Wimg.*(1 .- occlusion), img1, dims=4)


