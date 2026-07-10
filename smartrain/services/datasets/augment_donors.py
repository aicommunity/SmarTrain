"""Compatibility wrapper around dataset_augment."""

from __future__ import annotations

from smartrain.services.datasets import dataset_augment as _impl

_build_donor_pool = _impl._build_donor_pool
_match_patch_to_region = _impl._match_patch_to_region
_feather_alpha = _impl._feather_alpha
_pick_donor_balanced = _impl._pick_donor_balanced
_pick_donor_any = _impl._pick_donor_any
_pick_donor_soft = _impl._pick_donor_soft
_apply_copy_paste = _impl._apply_copy_paste

__all__ = [
    "_build_donor_pool",
    "_match_patch_to_region",
    "_feather_alpha",
    "_pick_donor_balanced",
    "_pick_donor_any",
    "_pick_donor_soft",
    "_apply_copy_paste",
]
