"""Shared predictor and authorized-residual preparation stage."""

from .fit_predictors import (
    fit_predictor_stage,
    load_authorized_labels,
    main,
    run_oracle_diagnostics,
    run_stage,
    verify_generator_inputs,
)

__all__ = [
    "verify_generator_inputs",
    "load_authorized_labels",
    "fit_predictor_stage",
    "run_oracle_diagnostics",
    "run_stage",
    "main",
]
