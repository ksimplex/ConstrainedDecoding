"""
twistroute: trajectory-exact constrained sampling for masked diffusion models.

The code that follows the notes (docs/refactor_map.md maps sections and equations to modules):

    constraints   section 2: the constraint interface of Proposition 1 and the families of Table 1
    state         the partly masked sequences x_t and the model's prediction at them
    twists        section 2.3: twists and proposals (product, eq. 9-10; chain, eq. 11-13); eq. (5)
    resampling    section 2.3: resampling rules
    sampler       section 2.3: the unconstrained step (eq. 4) and the weighted sampler (eq. 6)
    schedules     sections 1 and 4: unmasking schedules, sequential and parallel
    evaluation    Table 3, Figures 1-2: exact laws at short length, bounds, spread of Z_hat
    models        oracle models, real-model wrappers, token classes

The earlier packages, kept unchanged because other code imports them and because they hold the
options that were tried and set aside:

    compile       CPU reference algorithms for the constraint families (wrapped by `constraints`)
    torch_routes  their batched versions (wrapped by `constraints`)
    decode        the earlier sampler `twisted_smc` with all its options, policies, exact toy laws
    diagnostics   tail diagnostics, bootstrap intervals, hidden-mass decomposition
    experiment    shared plumbing of the earliest experiment scripts
"""
__version__ = "0.1.0"
