"""Seeding and run manifests.

seeding.py   set_seed(seed) -- called by every script before anything else.
manifest.py  RunManifest -- every run writes exactly one results/<ID>/manifest.json
             with the git commit, resolved config, seed, parameter counts, wall-clock,
             NFE, metrics, and library versions.

scripts/make_tables.py reads ONLY manifests. If a number is in the paper and not in a
manifest, it does not go in the paper.
"""
