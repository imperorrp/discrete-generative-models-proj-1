"""Training objectives.

flow_matching.py  x_t = (1-t) x_0 + t x_1 ;  target u_t = x_1 - x_0 ;  MSE
diffusion.py      x_k = sqrt(abar) x_1 + sqrt(1-abar) eps ;  target eps ;  MSE

Both apply condition dropout (p_drop) so the null-condition branch exists at sampling
time. Train BOTH models with dropout even though section 4.2 is unguided -- the branch
cannot be added afterwards.

Use the SAME loss reduction in both files. Mean-over-everything and sum-over-pixels
differ by a factor of the pixel count, which silently changes the effective learning
rate and breaks the matched comparison.
"""
