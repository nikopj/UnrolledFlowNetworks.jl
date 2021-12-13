# Unrolled Primal Dual Spliting for Optical Flow Estimation

## TODO
- cleanup:
	- separate modules
- write dataloader for flying things, middleburry (3)
- unroll TVL1\_VCA, TVL1\_FCA, image-driven smoothness prior (4)
	- sigmoid weighted smoothness
- verify @ein is not falling back to loop implementation on GPU
	- use tullio.jl?
- write visualization code (0)
	- intermediate flows
	- gt, flow\_hat, residual
- experiment with weight decay (1)
- get classical solver results (2)
	- grid-search hyperparams
