using UnrolledFlowNetworks
using Zygote, BenchmarkTools

net = BCANet(K=10, M=8, P=5, s=1; init=false)
ds = FlyingChairsDataset("dataset/FlyingChairs"; split="val", gray=true)
dl = Dataloader(ds, true; batch_size=4, crop_size=128)

