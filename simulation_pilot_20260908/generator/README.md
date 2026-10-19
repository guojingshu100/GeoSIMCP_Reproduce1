# S1--S4 generator `gccp_sim_s1_s4_v1`

This directory contains the frozen, paired data generator for the GCCP simulation pilot. It performs data generation and validation only; it does not fit GCCM or any conformal method.

From `F:\GeoSIMCP_Reproduce`, run:

```powershell
python .\simulation_pilot_20260908\generator\generate_scenarios.py --output-dir .\simulation_pilot_20260908
```

The command refuses to overwrite existing CSV files by default. To deliberately reproduce the frozen outputs in the same directory, add `--overwrite`. Reproducibility is checked internally by generating the eight CSV files in two separate temporary directories and comparing SHA-256 hashes.

The public API includes `sigmoid`, `scale_function`, `create_base_random_variables`, `create_frozen_split`, `generate_scenario`, `validate_scenario`, `write_outputs`, and `run_pipeline`.

Observed files contain only the approved model-visible fields. Latent `g_true`, `eta`, `epsilon`, `mu_true`, and `sigma_true` are kept in separate truth files for oracle diagnostics.
