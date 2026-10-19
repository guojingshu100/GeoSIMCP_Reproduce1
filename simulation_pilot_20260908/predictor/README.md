# Shared predictor stage `gccp_rf_predictor_v1`

This module fits one frozen `RandomForestRegressor` per S1--S4 scenario. Every scenario uses the same six raw observed features and model configuration. Labels from `final_calibration` and `test` are never loaded.

Run from `F:\GeoSIMCP_Reproduce`:

```powershell
python .\simulation_pilot_20260908\predictor\fit_predictors.py --working-dir .\simulation_pilot_20260908
```

The command verifies all generator hashes before fitting and refuses to overwrite existing stage outputs unless `--overwrite` is explicitly supplied. Truth data is parsed only inside the isolated oracle diagnostic after model, prediction, and residual outputs have been saved and hashed.

Frozen CSV predictions use a fixed `estimators_` order when averaging tree predictions. This removes last-bit variation from parallel floating-point aggregation while leaving the fitted `RandomForestRegressor(n_jobs=-1)` configuration unchanged.

This stage does not implement GCCM, distance learning, conformal calibration, intervals, or test evaluation.
