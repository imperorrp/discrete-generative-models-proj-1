"""Samplers and guidance rules. THE INNOVATION LIVES HERE.

samplers.py   euler (flow matching), ddim (diffusion). Both honest about NFE.
guidance.py   Every guidance rule behind one interface, so all of section 4.4 and 4.5
              runs from one code path and never forks per experiment:

                  none        w = 0, pure unconditional
                  fixed       standard CFG, constant w            (4.3 baseline)
                  interval    CFG only for t in [lo, hi]          (arXiv:2404.07724)
                  cfgpp       manifold-constrained CFG            (arXiv:2406.08070)
                  autoguid    guide with a weaker copy of itself  (arXiv:2406.02507)
                  adaptive    OURS: w predicted from h(c) and t

NFE accounting: N solver steps cost 2N function evaluations under ANY classifier-free
variant, because both branches are evaluated. Always run both branches, even at w = 1,
so the compute is matched across the sweep. Report the CFG-adjusted number.
"""
