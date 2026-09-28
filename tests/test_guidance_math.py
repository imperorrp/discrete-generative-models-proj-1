"""Dependency-free checks of the guidance math. Runs anywhere, no GPU, no data.

Invariants:
  CFG at w=0 returns the unconditional prediction; at w=1 the conditional one.
  The guided prediction is affine in w.
  The controller outputs exactly w_init at initialization.
  NFE: N solver steps cost 2N evaluations under any CFG variant.
"""
