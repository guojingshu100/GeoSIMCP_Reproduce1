# Stage4C independent-seed selector confirmation

Checkpoint: preliminary20; complete repeats by scenario {'S0': 20, 'S1': 20, 'S2': 20, 'S3': 20, 'S4': 20, 'S5': 20}; metric rows 820, single-mask rows 2200.
Formal claims use only the planned R=100 six-test primary family. R=20 is descriptive. No reused Stage4/4B samples, outcome arrays or performance files; only code/protocol and two explicit seed manifests.

## Selector candidate and interpretation
New HistGradientBoostingRegressor loss=quantile q=.9, 200 iterations, learning_rate .05, max_leaf_nodes15, min_leaf20, L2 .1, early_stopping False; full parameters in protocol. Target is observed Y upper quantile, not absolute RF residual. Fit D_fit1600, evaluate base/permuted pinball on D_loc800. Ten registered permutations per each of20 non-coordinate attributes, top-k exact with feature-name tie-break; no sign filter. Coordinates remain fixed. No D_cal/D_eval access during selection.
Common point predictor remains original RF300 fit on D_fit with all22 observed inputs. Selector only changes localization attributes. D_loc labels legitimately support both supervised selection and q0; independent D_cal is used only after selection/q0 freeze. This differs from the old unimplemented D_fit-internal residual selector and is not causal discovery.

## Primary Interval Score comparisons
Delta=GCCP-ResidualParent-RC minus comparator. One-sided paired t alternative mean delta<0. Holm over exactly six S2-S4 comparisons, FWER .05. Displayed95 CIs are unadjusted two-sided; no CI-of-coverage-difference equivalence claim.
| Scenario | Comparator | Score delta [t95] | Raw one-sided p | Holm p | Confirmatory rejection |
|---|---|---|---|---|---|
| S2 | RandomMask-k1-RC-Mean10 | -0.92216 [-1.02073, -0.82360] | 2.33715e-14 | 7.01146e-14 | False |
| S2 | SupervisedSelector-k1-RC | -0.98591 [-1.07983, -0.89200] | 2.86362e-15 | 1.14545e-14 | False |
| S3 | RandomMask-k1-RC-Mean10 | -0.85909 [-0.92823, -0.78995] | 1.284e-16 | 6.41999e-16 | False |
| S3 | SupervisedSelector-k1-RC | -0.97422 [-1.04627, -0.90217] | 2.6766e-17 | 1.60596e-16 | False |
| S4 | RandomMask-k1-RC-Mean10 | -0.84856 [-1.00194, -0.69519] | 2.35875e-10 | 2.35875e-10 | False |
| S4 | SupervisedSelector-k1-RC | -0.84009 [-0.97318, -0.70700] | 2.51006e-11 | 5.02011e-11 | False |

## Absolute coverage and efficiency
| Scenario | Method | Coverage [t95] | Width | Interval Score | Coverage vs .90 |
|---|---|---|---|---|---|
| S0 | GCCP-OracleParent-RC | 0.90225 [0.89685, 0.90765] | 3.40341 | 4.24369 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S0 | GCCP-ResidualParent-RC | 0.90225 [0.89685, 0.90765] | 3.40341 | 4.24369 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S0 | GeoSIMCP-RC | 0.90225 [0.89685, 0.90765] | 3.40341 | 4.24369 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S0 | RandomMask-k0-RC-Mean10 | 0.90225 [0.89685, 0.90765] | 3.40341 | 4.24369 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S0 | SupervisedSelector-k0-RC | 0.90225 [0.89685, 0.90765] | 3.40341 | 4.24369 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | GCCP-OracleParent-RC | 0.90056 [0.89377, 0.90736] | 3.42371 | 4.31653 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | GCCP-ResidualParent-RC | 0.89969 [0.89097, 0.90841] | 3.45788 | 4.37221 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | GeoSIMCP-RC | 0.89969 [0.89097, 0.90841] | 3.45788 | 4.37221 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | RandomMask-k0-RC-Mean10 | 0.89969 [0.89097, 0.90841] | 3.45788 | 4.37221 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | RandomMask-k1-RC-Mean10 | 0.90099 [0.89417, 0.90782] | 3.42112 | 4.31259 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | SupervisedSelector-k0-RC | 0.89969 [0.89097, 0.90841] | 3.45788 | 4.37221 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S1 | SupervisedSelector-k1-RC | 0.90056 [0.89377, 0.90736] | 3.42371 | 4.31653 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | GCCP-OracleParent-RC | 0.89712 [0.88839, 0.90586] | 5.52822 | 7.17588 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | GCCP-ResidualParent-RC | 0.89787 [0.88880, 0.90695] | 5.50903 | 7.10483 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | GeoSIMCP-RC | 0.89931 [0.89013, 0.90850] | 5.65049 | 7.68953 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | RandomMask-k1-RC-Mean10 | 0.89461 [0.88710, 0.90211] | 5.71119 | 8.02699 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | RandomMask-k2-RC-Mean10 | 0.89525 [0.88738, 0.90312] | 5.77044 | 8.06095 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | SupervisedSelector-k1-RC | 0.89362 [0.88632, 0.90093] | 5.72202 | 8.09074 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S2 | SupervisedSelector-k2-RC | 0.89712 [0.88839, 0.90586] | 5.52822 | 7.17588 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | GCCP-OracleParent-RC | 0.90281 [0.89677, 0.90885] | 5.60183 | 7.17668 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | GCCP-ResidualParent-RC | 0.90394 [0.89806, 0.90981] | 5.59679 | 7.08231 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | GeoSIMCP-RC | 0.90056 [0.89580, 0.90532] | 5.58716 | 7.43152 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | RandomMask-k1-RC-Mean10 | 0.89964 [0.89402, 0.90527] | 5.80814 | 7.94140 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | RandomMask-k2-RC-Mean10 | 0.90009 [0.89519, 0.90500] | 5.80431 | 7.86831 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | SupervisedSelector-k1-RC | 0.90006 [0.89454, 0.90558] | 5.81238 | 8.05653 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S3 | SupervisedSelector-k2-RC | 0.90281 [0.89677, 0.90885] | 5.60183 | 7.17668 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S4 | GCCP-OracleParent-RC | 0.90281 [0.89906, 0.90657] | 10.13246 | 13.74092 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S4 | GCCP-ResidualParent-RC | 0.90431 [0.90032, 0.90830] | 10.08308 | 13.61200 | CI_ENTIRELY_ABOVE_90 |
| S4 | GeoSIMCP-RC | 0.90387 [0.89904, 0.90871] | 10.36460 | 14.25459 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S4 | RandomMask-k1-RC-Mean10 | 0.90587 [0.90114, 0.91060] | 10.40511 | 14.46057 | CI_ENTIRELY_ABOVE_90 |
| S4 | RandomMask-k2-RC-Mean10 | 0.90519 [0.90072, 0.90965] | 10.47298 | 14.51457 | CI_ENTIRELY_ABOVE_90 |
| S4 | SupervisedSelector-k1-RC | 0.90794 [0.90353, 0.91235] | 10.39848 | 14.45209 | CI_ENTIRELY_ABOVE_90 |
| S4 | SupervisedSelector-k2-RC | 0.90281 [0.89906, 0.90657] | 10.13246 | 13.74092 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S5 | GCCP-OracleParent-RC | 0.90775 [0.90200, 0.91350] | 5.58333 | 6.96630 | CI_ENTIRELY_ABOVE_90 |
| S5 | GCCP-ResidualParent-RC | 0.90575 [0.90010, 0.91140] | 5.54577 | 6.87310 | CI_ENTIRELY_ABOVE_90 |
| S5 | GCCP-WrongPrior-RC | 0.90981 [0.90361, 0.91602] | 5.88217 | 7.91049 | CI_ENTIRELY_ABOVE_90 |
| S5 | GeoSIMCP-RC | 0.90569 [0.89984, 0.91154] | 5.68112 | 7.46855 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S5 | RandomMask-k1-RC-Mean10 | 0.90618 [0.89945, 0.91291] | 5.78802 | 7.85796 | CI_INCLUDES_90_NOT_ESTABLISHED |
| S5 | RandomMask-k2-RC-Mean10 | 0.90637 [0.90019, 0.91256] | 5.83730 | 7.85111 | CI_ENTIRELY_ABOVE_90 |
| S5 | SupervisedSelector-k1-RC | 0.90744 [0.90086, 0.91402] | 5.81573 | 7.86505 | CI_ENTIRELY_ABOVE_90 |
| S5 | SupervisedSelector-k2-RC | 0.90775 [0.90200, 0.91350] | 5.58333 | 6.96630 | CI_ENTIRELY_ABOVE_90 |
A CI including .90 is compatible with the target but does not establish coverage>=.90. A CI wholly below .90 flags undercoverage. One-sided lower>=.89 is only the separately reported practical screen. Score-test rejection is not automatically a coverage-qualified efficiency or causal-identification conclusion.

## Selection frequencies (developer labels only)
| Scenario | k | Feature | Selected / repeats |
|---|---|---|---|
| S0 | 0 | EMPTY_FALLBACK | 20 / 20 |
| S1 | 0 | EMPTY_FALLBACK | 20 / 20 |
| S1 | 1 | X1 | 20 / 20 |
| S2 | 1 | X1 | 20 / 20 |
| S2 | 2 | X1 | 20 / 20 |
| S2 | 2 | X2 | 20 / 20 |
| S3 | 1 | X1 | 20 / 20 |
| S3 | 2 | X1 | 20 / 20 |
| S3 | 2 | Z | 20 / 20 |
| S4 | 1 | X1 | 20 / 20 |
| S4 | 2 | X1 | 20 / 20 |
| S4 | 2 | X2 | 20 / 20 |
| S5 | 1 | X1 | 20 / 20 |
| S5 | 2 | X1 | 20 / 20 |
| S5 | 2 | X2 | 20 / 20 |

All predeclared masks retained; ten metrics averaged within replicate, no mask-level pseudoreplication. Max infinite fraction 0.0; finite-conditional coverage and finite fractions are in metrics and summaries. No difficult-query removal.
S0 validates exact ordinary fallback; all-k0 skips an irrelevant selector fit. S1 residual empty means ordinary GeoSIM fallback, not CP or an oracle-mean predictor. S5 correct full/residual parents and wrong {X1,X3} are distinct methods.

## Limits of the conclusion
Oracle residual sets are known SCM scale drivers, not proven full parents of fitted RF residuals. Beating this observed-Y upper-quantile selector can reflect better alignment with residual uncertainty; it does not show superiority over all supervised residual-aware selectors. Full-parent/other-scenario/width/feature-frequency analyses are secondary descriptive and not covered by the six-test Holm claim. No real-data causal or dependent-spatial validation.

## Reproduction
From project root: python experiments/gccp_stage4c_selector_confirmation_20261019_v1/experiment.py prepare; smoke --workers6; batch --start0 --stop20 --workers6; report --tag preliminary20; batch --start20 --stop100 --workers6; report --tag final100. Supply spaces between options and numbers (e.g. --workers 6). Prepare refuses overwrite. Exact repeat uses the recorded source, software, namespace and roots in a fresh directory.
