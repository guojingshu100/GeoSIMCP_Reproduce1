# Method provenance

## Geographical cross mapping cardinality (GCMC)

- Paper: Lyu et al. (2026), *Measuring causal strengths from spatial cross-sectional data with geographical cross mapping cardinality*.
- DOI: `10.1080/13658816.2026.2687121`.
- Official repository: [SpatLyu/GCMC_IJGIS](https://github.com/SpatLyu/GCMC_IJGIS).
- Pinned upstream commit: `4b1c6543fbcbdc67516fb27b2cdac8518af65386`.
- Relevant upstream workflow: `Synthetic benchmark/Synthetic benchmark.r`, supplied SHA-1 `2469252e8eacd072276159bd746158f2def1f0b5`.
- Official API used for integration: `spEDM::gcmc()` and `spEDM::fnn()`; no paper-equation reimplementation is used.
- Upstream reported reproduction environment: Windows 11, R 4.5.3, spEDM 1.12.
- Local Stage-10A environment (2026-09-17): Windows 11, R 4.6.1, Rtools45 revision 6768, and project-local `spEDM` 1.12. The installed runtime passed a separate clean-session verification. The build-time repair is recorded under `experiments/causal_strength_diagnostic/20260917_stage10a_env_spedm112_v10_deep_repair/`.
- Local Stage-10B adaptation (2026-09-17): seed-20270908 frozen scale-fit rows only; `A_model` constructed from frozen predictions; an explicit Delaunay neighbor list derived from the frozen 200 scale-fit coordinates; `x1`, `x2`, `g_proxy`, and `x_noise` tested as causes of `A_model`; coordinates define geometry only. The actual calls use installed `spEDM::gcmc()` and `spEDM::fnn()`.
- Paper reproduction-repository license: UNVERIFIED. No license is asserted for `SpatLyu/GCMC_IJGIS`, and no source from that repository was copied into the local implementation.
- Modification status: no changes to spEDM or the upstream GCMC algorithm. Local code performs artifact validation, explicit-neighbor construction, API orchestration, and result extraction only.

## spEDM package (pinned GCMC runtime)

- Package repository: [stscl/spEDM](https://github.com/stscl/spEDM).
- Required release tag: `v1.12`.
- Required release commit: `43c5c136ae909c5de83535438d5d5953001429c5`.
- Package license: GPL-3.
- Intended API: `spEDM::gcmc()` and `spEDM::fnn()`.
- Stage 10A-ENV status (2026-09-17): spEDM 1.12 was compiled from the verified local CRAN source archive into `environment/gcmc_spedm_1.12_r4.6.1/library`. No global R library was modified. The build-time issue was classified as `MAKE_SHELL_PATH_PROPAGATION`; the installed runtime does not require the build override.
- This GPL-3 statement applies to the `spEDM` package only. The license of `SpatLyu/GCMC_IJGIS` remains UNVERIFIED.
- Installed runtime source: local CRAN `spEDM_1.12.tar.gz`, SHA-256 `70dad0f8d41a4a31d0d4cb3b74273e2034dd87227d85abcaa8eb4c927e4cc153`. This is recorded separately from the known upstream GitHub v1.12 commit; byte identity is not asserted.
