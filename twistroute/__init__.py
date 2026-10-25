"""
twistroute: exact constrained sampling for masked diffusion language models with route-generic
twisted SMC.

    compile       CPU reference routes (automata, grammar charts, factor graphs, spanning trees)
    torch_routes  batched GPU routes: logZ, exact sampling, membership
    decode        twisted SMC (TWISTER generalised to any route), native sampling, exact toy laws
    models        denoiser wrappers and token-class projections
    diagnostics   tail diagnostics, bootstrap CIs, pairwise TV with matched floor, hidden-mass decomposition
    experiment    shared plumbing for the experiment scripts (loading, batched runs, native counts)
"""
__version__ = "0.1.0"
