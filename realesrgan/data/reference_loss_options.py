"""Backward-compatible reference-loss switch parsing for isolated Y4UV training."""

from __future__ import annotations

from typing import Any, Mapping


def _coerce_bool_option(value: Any, name: str, default: bool = True) -> bool:
    """@shanlan-comment Normalize one config boolean while retaining the historical true default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


def resolve_reference_loss_flags(train_opt: Mapping[str, Any]) -> tuple[bool, bool]:
    """@shanlan-comment Resolve train/validation switches with legacy-true defaults."""
    reference_option = train_opt.get("reference_loss")
    if isinstance(reference_option, Mapping):
        shared_default = reference_option.get("enabled")
        train_reference_default = reference_option.get(
            "train",
            True if shared_default is None else shared_default,
        )
        validation_reference_default = reference_option.get(
            "validation",
            reference_option.get(
                "validation_enabled",
                True if shared_default is None else shared_default,
            ),
        )
    elif reference_option is None:
        train_reference_default = True
        validation_reference_default = True
    else:
        train_reference_default = reference_option
        validation_reference_default = reference_option
    train_reference_value = train_opt.get(
        "train_reference_loss",
        train_opt.get(
            "reference_loss_enabled",
            train_opt.get("compute_reference_loss", train_reference_default),
        ),
    )
    validation_reference_value = train_opt.get(
        "validation_reference_loss",
        train_opt.get(
            "reference_loss_validation",
            validation_reference_default,
        ),
    )
    return (
        _coerce_bool_option(train_reference_value, "train.train_reference_loss"),
        _coerce_bool_option(validation_reference_value, "train.validation_reference_loss"),
    )


__all__ = ["resolve_reference_loss_flags"]
