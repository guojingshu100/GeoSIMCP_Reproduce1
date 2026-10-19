# Causal-only conformal prediction: Stage-1 audit

**Audit date:** 2026-09-17 (Asia/Shanghai)  
**Canonical root:** `F:\GeoSIMCP_Reproduce`  
**Audit mode:** read-only with respect to experimental implementation  
**Artifact created by this stage:** this report only

No experiment, model fit, conformal evaluation, test suite, notebook, or R job was run for this audit. Stable code and existing results were not modified or moved. The audit used directory/file inspection, notebook JSON inspection, CSV/JSON reads, and checksum recomputation.

## 1. Executive decision

The repository is **not ready for a confirmatory causal-only CP experiment or a causal-performance claim**. It is **conditionally ready for a new, versioned implementation stage** after the method and validity contract are frozen.

The decisive findings are:

1. There is no dedicated immutable causal-only experiment folder, run command, configuration, log, environment snapshot, or test result. Existing `lambda=0` rows are boundary cases inside broader historical tuning grids. They are useful audit evidence, not a completed causal-only experiment.
2. The current residual-target GCCM estimator fails the most important simulation face-validity check. In S2, where `x2` is the true noise-scale driver, learned GCCM gives `x2` zero weight and instead weights `s_x`, `s_y`, and `g_proxy`. In homoscedastic S4 it again assigns all weight to `s_x` and `s_y`.
3. The S4 permutation control shows that spatial GCCM convergence is common under this null. The observed S4 spatial statistics are not extreme relative to the permutation reference. These weights cannot presently be interpreted as reliable causal-strength estimates.
4. The CMI diagnostics show that an oracle S2 scale signal is detectable over repeated samples, but the corresponding repeated **model-residual** Stage 8 has source code only and no output. Therefore the learned target needed by an operational method is not yet established.
5. Historical Seattle and PM2.5 notebooks use calibration-only weighted quantiles without the test-point `+infinity` atom. The newer package contains a corrected engine, but the legacy wrappers default to the historical behavior. A new experiment must make this choice explicit and cannot inherit the notebook implementation silently.
6. The existing Seattle and PM2.5 test labels have already been opened and their results are in the repository. They are not untouched confirmatory test sets for a newly introduced causal-only method.

**Recommended gate:** implement and validate causal-only CP first on the frozen simulation framework. Treat GCCM-weighted causal-only CP as a diagnostic/negative-control branch unless a revised causal-strength estimator passes the S2/S4 identification gates. Do not run final calibration or test evaluation until the Stage-2 acceptance criteria in Section 12 are met.

## 2. Meaning of “causal-only” that should be frozen

The repository currently uses `lambda` with different but compatible fusion formulas. For additive GCCP-B,

`W = lambda * K_geo + (1 - lambda) * K_causal`.

Thus `lambda=0` means feature/causal-kernel only and `lambda=1` means geography only. For a new experiment, use an explicit method name and do not rely on endpoint semantics alone.

Recommended definition:

> **CausalOnly-NormCP:** no geographic distance or geographic kernel enters the interval. Observed features are standardized using only authorized scale-fit/calibration rows. A pre-frozen causal-strength rule defines a feature metric, which is used to estimate a local residual scale. A separate final-calibration split conformalizes normalized residuals.

For nonnegative normalized strengths `w_j` over `p` observed features, the existing scale-preserving metric is

`gamma_j = p * w_j`

and

`d_causal(x, x') = sqrt(sum_j gamma_j * (z_j - z'_j)^2)`.

Equal strengths then reduce exactly to standardized Euclidean distance. This is an essential unit check.

The label “causal” must remain qualified. GCCM or CMI-derived feature weights are causally motivated estimators under their assumptions; inserting them into a distance does not by itself make the resulting conformal interval causally identified or distribution-free. Reports should state the assumed direction, target, conditioning set, graph/embedding construction, and limitations.

Two architectures must not be conflated:

- **Direct local weighted quantile:** `K_causal` directly weights calibration residuals for each prediction point. This is what the historical `lambda=0` GCCP rows represent.
- **Normalized split CP:** `d_causal` estimates a positive local scale, while an independent final-calibration split supplies a single finite-sample conformal quantile. This is the architecture already preregistered in the simulation protocol and is the recommended primary architecture.

## 3. Relevant repository inventory

| Component | Evidence | Audit status |
|---|---|---|
| Frozen S1–S4 generator | `simulation_pilot_20260908/generator/config_v1.json`, generator source, checks | Available; `n=2000`, paired base variables, seed and split scheme recorded |
| Predictor/residual pipeline | `simulation_pilot_20260908/predictor/` and `predictor_outputs/` | Available; train-only predictor and authorized residual artifacts exist |
| Scale-comparison protocol | `simulation_pilot_20260908/01_pilot_protocol_draft.md`, `scale_comparison/` | Strong design basis; historical status documents are partly superseded by later runtime work |
| Residual-target GCCM | `simulation_pilot_20260908/gccm_runtime/`, `gccm_outputs/S1`–`S4` | Executed and reproducibly recorded, but scientifically fails the key S2/S4 identification check |
| GCCM null control | `simulation_pilot_20260908/gccm_negative_control/S4_effect_permutation_v1/` | 99 permutations complete; strongly cautions against interpreting spatial convergence as causal strength |
| CMI adapter and diagnostics | `causal_strength_diagnostic_v1/` | Stages 3–7 recorded; Stage 8 source exists but Stage 8 outputs do not |
| Finite-sample conformal engine | `geoconformal-main/geoconformal/geocp/` | Corrected test-atom support and tests exist |
| Historical Seattle causal weighting | Seattle `03_gccp` and `04_gccp` notebooks/results | Exploratory historical evidence; hard-coded paths and notebook execution |
| Historical PM2.5 causal weighting | PM2.5 `03_gccm`, `04_gccp`, manifests/results | Best preserved real-data workflow, but outcome-target rather than residual-scale-target GCCM |
| Dedicated causal-only experiment | Expected under `experiments/causal_only_cp/<dated_name>/` | **Absent** |
| Method provenance register | `docs/methodology/METHOD_PROVENANCE.md` | **Absent** |

The canonical root itself is not a Git repository and lacks root-level `README.md`, `LICENSE`, and `.gitignore`. A license and README exist under `geoconformal-main`, but there is no `.git` history there either. Consequently, existing artifacts have hashes and dates but no repository commit identity.

## 4. Scientific audit of the causal-strength signal

### 4.1 Generator truth and authorized splits

The frozen generator defines:

- S1: `mu=2*x1`, `sigma=h(x1)`;
- S2: `mu=2*x1`, `sigma=h(x2)`;
- S3: spatial common driver `g_true`, observed through noisy `g_proxy`;
- S4: `mu=2*x1`, constant `sigma=1.3`.

All base variables, including coordinates and `x2`, are mutually independent. The split is `train/local_calibration/validation/final_calibration/test = 1000/300/200/200/300`, with local calibration divided into `scale_fit=200` and `tuning_calibration=100`. This is suitable for leakage-controlled method development.

### 4.2 Residual-target GCCM does not recover the S2 driver

Existing `gccm_outputs/S2/weights.csv` gives:

| Feature | Learned weight |
|---|---:|
| `s_x` | 0.477051 |
| `s_y` | 0.287773 |
| `g_proxy` | 0.235175 |
| `x1` | 0 |
| `x2` | **0** |
| `x_noise` | 0 |

This is opposite to the known S2 scale mechanism. The Stage-2 GCCM audit explains why this can happen: the graph-ring embedding contains neighboring residual summaries, while S2's same-location `x2 -> sigma` relationship is nonspatial. The focal `x2` signal is not naturally represented by the spatial effect-manifold embedding.

In S4, the method assigns `0.454343` to `s_x` and `0.545657` to `s_y`, with all other features zero. The 99-permutation control reports spatial convergence in 64.6% of permutations for `s_x` and 70.7% for `s_y`; the observed terminal-rho tail probabilities are 0.45 and 0.33. The audit interpretation is limited to this implementation and DGP, but it is sufficient to reject these weights as the primary learned causal metric for the proposed experiment.

### 4.3 CMI detects oracle signal, but the operational target remains incomplete

At the frozen single replicate, refined CMI results are weak/mixed: model-residual S2 M0 `p=0.066`; oracle S2 M0 `p=0.058`; oracle S2 conditioned on space `p=0.039`. These should not be read as a clean single-seed success.

The repeated 20-replicate oracle pilot is more encouraging:

- S2 rejection rate: 0.90 for M0 and 0.90 for Mspace;
- S4 rejection rate: 0.00 for M0 and 0.05 for Mspace;
- S2 raw score exceeded S4 in all 20 paired replicates for both conditioning choices.

However, this target is `A_oracle=|y-mu_true|`. A deployable experiment uses `A_model=|y-y_pred|`. `run_stage8_model_residual_power_pilot_v1.py` is present, but neither `stage8_model_residual_replicates.csv` nor `stage8_model_residual_summary.json` exists. Stage 8 therefore remains **not run**, and this audit did not run it.

There is also no frozen mapping from possibly negative/noisy CMI estimates and permutation results to nonnegative distance weights. CMI evidence cannot yet be substituted for a fully specified CausalCP metric.

### 4.4 Latest scale-learning gate

The most recent gate file classifies both S2 and S4 as `SCALE NOT ESTABLISHED`. An earlier D7A diagnostic classified an S2 KNN result as confirmed, while D7B classified the S4 negative control as a failure. These artifacts answer narrower questions and should not be collapsed into a positive end-to-end conclusion. The conservative current status is that local scale learning is unresolved.

## 5. What historical `lambda=0` results do and do not show

These values were read from existing files; they were not regenerated.

| Dataset/artifact | Causal-only boundary result | Interpretation |
|---|---|---|
| Seattle additive GCCP validation | `lambda=0`, `h_feature=3.027984`, coverage `0.906667`, mean interval `0.242576`, interval score `0.317218`, median `n_eff=265.54` | Feasible validation boundary point. Not a standalone frozen experiment and not a causal identification result. The unused `h_geo` column should be recorded as NA at this boundary. |
| PM2.5 GCCP-B validation | `lambda=0`, `h_causal=1.534340`, coverage `0.845070`, mean width `17.204449`, interval score `31.845347`, median `n_eff=79.68` | Misses the 0.90 target. The selected mixed method was `lambda=0.1`, also infeasible at coverage `0.852113`; selection fell back to global minimum interval score. |
| PM2.5 final test | No causal-only row | The final table evaluates selected GCCP-B (`lambda=0.1`), not `lambda=0`. A causal-only final-test claim cannot be reconstructed from this summary. |

The real-data GCCM target is another critical boundary. PM2.5 weights were learned from detrended PM2.5 and Seattle weights from the corresponding outcome construction, not from prediction-error magnitude or a local noise scale. They may be used as outcome-target secondary diagnostics, but they do not answer whether a learned causal metric localizes residual uncertainty.

## 6. Conformal-validity audit

The historical Seattle notebook defines an author-style weighted quantile that normalizes calibration weights only. The PM2.5 workflow uses `tau=128/141=0.907801...`, which gives the standard split-CP rank for uniform weights but still omits a weighted test-point atom. Neither is the same as the corrected weighted construction for nonuniform local weights.

The newer `geoconformal.geocp` engine supports a test-point atom at `+infinity` with self-weight `w(x)` and uses a step quantile. Its uniform-weight behavior is tested against textbook split CP. Important caveats remain:

- `GeoConformalPrediction.conformalize` defaults to `include_test_atom=True`, but the legacy public GeoCP/GeoSIMCP wrappers default to `False` for numerical backward compatibility.
- Adding the atom fixes the finite-sample accounting but does not make an arbitrary spatial or learned similarity kernel distribution-free. A guarantee still requires an appropriate weighted-exchangeability/covariate-shift argument, or the result must be presented as an empirically evaluated localized method.
- Learned weights and hyperparameters must be label-independent for the final-calibration/test pair, or be learned on an earlier disjoint split exactly as preregistered.

For the recommended normalized architecture, the safest primary claim is the ordinary split-conformal marginal guarantee conditional on a scale function frozen before final calibration, under exchangeability of the normalized final-calibration and test scores. Conditional/local coverage should be reported diagnostically, not claimed automatically.

## 7. Leakage and test-set audit

Positive findings:

- Simulation predictor fitting is train-only.
- GCCM/CMI diagnostic loaders restrict themselves to `scale_fit=200` and record that validation/final-calibration/test rows were not accessed.
- Seattle uses inner fit/calibration/validation roles for tuning before its outer evaluation.
- PM2.5 records `990/141/142/142` train/calibration/validation/test counts, uses train-only GCCM weights, fits the similarity scaler on calibration only, freezes a manifest, constructs intervals before reading test labels, and records one-shot completion.
- The PM2.5 semantic split hash recomputes exactly as `785436f1...ddf04f3`. Its manifest hash recomputes as `d2090944...d7553` under the notebook's text/newline-normalized hashing convention. Raw file-byte hashes differ because they answer a different question; future manifests should label canonical-content hashes versus byte hashes explicitly.

Limitations:

- The Seattle and PM2.5 frozen tests are already visible. Any new method designed after inspection requires a fresh external holdout, a new preregistered split that does not reuse exposed labels, or nested/cross-fitted evaluation described as retrospective rather than confirmatory.
- Several historical notebooks embed `E:\GeoSIMCP_Reproduce` and machine-specific Python/R paths. They do not run from the current `F:` root without edits.
- Notebook execution order is the effective run script for major real-data workflows; exact shell commands and immutable environment snapshots are not colocated with the results.

## 8. Reproducibility completeness for a causal-only experiment

| Requirement | Current status |
|---|---|
| Dedicated source code | Missing; only endpoint logic embedded in notebooks and related reusable components |
| Immutable experiment directory | Missing |
| Single frozen config | Missing |
| Exact run command | Missing |
| Seeds | Present in source artifacts (`42`, generator seeds, CMI seeds), but not unified for causal-only CP |
| Input identifiers/checksums | Partial; simulation hashes are strong, real-data manifests are incomplete |
| Split identifiers | Strong for simulation and PM2.5; Seattle split arrays exist but lack a colocated causal-only manifest |
| Preprocessing/scaling | Defined in notebooks/protocol; not yet frozen in one causal-only config |
| Causal-strength estimator | GCCM fully parameterized but unsuitable for S2 primary use; CMI operational mapping incomplete |
| Conformal parameters | Historical values exist; the new method's validity mode is not frozen |
| Hyperparameters/search budget | Partial; normalized KNN grid is preregistered, CMI-to-weight and causal-only settings are not |
| Environment/package versions | Multiple snapshots exist but conflict by stage; no causal-only environment file |
| Metrics/tables/figures | Historical boundary metrics exist; no new causal-only outputs |
| Warnings/errors/logs | Strong for simulation GCCM; weak for notebook-only real-data runs |
| Method explanation/interpretation | Distributed across reports; no causal-only README/method notes |

Audit-time environment was Windows 11, Python 3.13.5 at `D:\conda\python.exe`; `Rscript` was not on `PATH`. Historical notebooks report Python 3.10.20. Simulation records also contain Python 3.13.5 environments, and the GCCM runtime records R 4.6.1 plus package versions. These are stage-specific environments, not one reproducible causal-only environment.

## 9. Data-publication audit

Current byte-level SHA-256 values observed during this audit are:

- PM2.5 CSV: `eeae7b3801bf11d4a243ac40ce70f16a5ddfa07ce9c343a1864f4d54b77221c9`;
- Seattle sample CSV: `33fe6a413827718a5a209312e24a804ab6c8402ba86cd0ad66bc0c36e53d8c8d`.

These identify local files but do not establish redistribution rights. The repository does not provide a complete dataset manifest containing source, download URL/DOI, license, retrieval date, checksum convention, and preprocessing history for either dataset. Until that exists, neither raw file should be assumed safe for a public GitHub release.

## 10. Method-provenance and publication audit

`docs/methodology/METHOD_PROVENANCE.md` is absent. Because this stage permits only the requested audit file, it was not created separately here. Before implementation, it must record at least:

- GeoCP/GeoSIMCP paper and code provenance; the local package declares MIT and `https://github.com/pengtum/geoconformal`, but no source commit is recoverable from the current directory;
- GCCM paper title, DOI, repository/release/commit, origin of `GCCM4Lattice.r`, relevant functions, and its license;
- Tigramite/CMIknn version, documentation or paper, relevant class/methods, and the local adapter modifications;
- all project modifications: graph construction, direction `effect_xmap_cause`, convergence criteria, strength normalization, distance transformation, test-atom behavior, and split restrictions.

No obvious credential-like filenames were found in the scoped inventory. This is not a content-level secret scan. Publication hygiene is currently weak because there is no root `.gitignore`, and the tree contains virtual environments, `__pycache__`, build/dist artifacts, installer binaries, notebook checkpoints, and machine-specific paths.

## 11. Recommended Stage-2 experiment design (not executed)

Create a new immutable folder such as:

`experiments/causal_only_cp/20260917_simulation_causal_only_cp_v1/`

Required comparisons:

1. `Raw-CP`;
2. `Normalized-Global` as an algebraic/control baseline;
3. `Normalized-EUC` with the same KNN scale estimator and search budget;
4. `CausalOnly-Oracle` using the true scale driver, clearly labeled as an upper-bound diagnostic;
5. `CausalOnly-Learned` only after its estimator passes the gates below;
6. residual-target spatial GCCM as a secondary diagnostic/negative control, not the primary learned method under the current evidence.

Reuse the frozen simulation roles and preserve the existing access order:

1. fit the predictor on `train=1000`;
2. learn preprocessing, causal strengths, and scale model on `scale_fit=200`;
3. form provisional normalized scores on `tuning_calibration=100`;
4. select from a preregistered, equal-budget grid on `validation=200`;
5. freeze code/config/hashes;
6. estimate the final normalized conformal order statistic on `final_calibration=200` (`k=181` for `alpha=0.1`);
7. construct all intervals before opening `test=300` labels;
8. save per-sample and aggregate metrics, logs, warnings, environment, and interpretation.

Use the corrected split-conformal implementation or an independently tested equivalent. If direct causal-kernel weighted quantiles are retained as an ablation, record `include_test_atom`, self-weight, zero-weight/underflow handling, infinite-threshold rate, effective sample size, and the precise theoretical claim.

## 12. Stage-2 acceptance criteria

Implementation should not proceed to final calibration/test unless all are true:

1. **Definition gate:** the primary architecture, estimand, causal direction, target (`A_model` versus oracle), conditioning set, and CMI/GCCM-to-weight transform are frozen before validation.
2. **Signal gate:** over preregistered paired replicates, the learned method distinguishes S2 from S4 using model residuals, not only oracle residuals; uncertainty intervals for rejection rates/effect contrasts are reported.
3. **Feature-recovery gate:** S2 assigns meaningful influence to `x2`; S4 does not repeatedly invent a concentrated causal metric. Exact thresholds must be preregistered.
4. **Degeneracy gate:** equal causal weights reproduce standardized EUC exactly; zero geography is verified numerically; inactive hyperparameters are saved as NA.
5. **Leakage gate:** automated access logs confirm no validation labels during causal-strength/scale fitting and no final/test labels before their authorized stages.
6. **Conformal gate:** uniform weights reproduce the exact split-CP order statistic; test-atom and self-weight behavior are explicitly tested for any direct weighted method.
7. **Stability gate:** all scales/weights are finite, scales are positive, underflow and infinite intervals are reported, and results are reproducible from a clean output directory.
8. **Negative-control gate:** S4 performance is not presented as causal localization evidence; any apparent gain triggers the frozen null/permutation interpretation.
9. **Reproducibility gate:** the experiment folder contains `README.md`, `config.json`, `run.ps1` or `run.py`, `metrics.csv`, `run_log.txt`, `environment.txt`, `method_notes.md`, input/split hashes, and exact command.
10. **Provenance gate:** method and dataset provenance records are complete enough for a public repository.

## 13. Final audit interpretation

The project has unusually rich intermediate diagnostics, frozen split work, and leakage-aware simulation infrastructure. Those assets make a rigorous causal-only experiment feasible. They do not yet establish that the current learned causal metric works.

The strongest current evidence is a contrast: CMI can recover the oracle S2 dependence over repeated samples, while the spatial GCCM residual-target implementation does not recover `x2` and produces spatial weights under S4. This points to an estimator/representation problem, not evidence that causal information is useless. The next scientific step is therefore to validate a fully specified model-residual causal-strength estimator under S2/S4 before any causal-only conformal performance experiment is allowed to consume final-calibration or test labels.

