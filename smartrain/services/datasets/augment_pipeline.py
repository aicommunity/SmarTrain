"""Compatibility wrapper around dataset_augment."""

from __future__ import annotations

from smartrain.services.datasets import dataset_augment as _impl

_split_images_rel = _impl._split_images_rel
augment_output_structure = _impl.augment_output_structure
_swap_images_labels_rel = _impl._swap_images_labels_rel
_augment_ensure_base_dirs = _impl._augment_ensure_base_dirs
_augment_output_paths = _impl._augment_output_paths
_write_data_yaml = _impl._write_data_yaml
_update_datasets_sidecar = _impl._update_datasets_sidecar
_list_workspace_detector_models = _impl._list_workspace_detector_models
_augment_roi_prompt_label = _impl._augment_roi_prompt_label
_print_augment_placement_mode_help = _impl._print_augment_placement_mode_help
_default_placement_mode_for_interactive = _impl._default_placement_mode_for_interactive
_augment_balancing_block_title = _impl._augment_balancing_block_title
_interactive_fill = _impl._interactive_fill

__all__ = [
    "_split_images_rel",
    "augment_output_structure",
    "_swap_images_labels_rel",
    "_augment_ensure_base_dirs",
    "_augment_output_paths",
    "_write_data_yaml",
    "_update_datasets_sidecar",
    "_list_workspace_detector_models",
    "_augment_roi_prompt_label",
    "_print_augment_placement_mode_help",
    "_default_placement_mode_for_interactive",
    "_augment_balancing_block_title",
    "_interactive_fill",
]
