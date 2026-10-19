# Residual-scale comparison stage

Current stage version: `gccp_residual_scale_v1_blocked`.

The GCCM applicability audit found that the historical irregular-point workflow is conceptually usable, but the learned branch cannot currently run: `Rscript` is unavailable and the GCCM library-size grid for only 200 scale-fit sites has not been preregistered. See `00_gccm_applicability_audit.md`.

`fit_local_scales.py` therefore performs shared non-GCCM preparation only. It:

- verifies the frozen generator/predictor status and the safe predictor-output hashes;
- loads observed features without loading `y`, restricted to the 200 `scale_fit` rows;
- loads only the 200 `scale_fit` values of `abs_residual`;
- fits one shared-form `StandardScaler` per scenario on `scale_fit` only;
- prepares deterministic leave-one-out EUC KNN scales for the frozen k grid;
- records Raw-CP and Normalized-Global scale constants;
- does not inspect tuning-calibration or validation labels and does not run a comparison.

Run from the project root:

```powershell
python .\simulation_pilot_20260908\scale_comparison\fit_local_scales.py --working-dir .\simulation_pilot_20260908
```

The entry point refuses to overwrite existing outputs unless `--overwrite` is explicitly supplied. It never reads or hashes truth files or the label-bearing generator CSVs as whole files.
