"""Compatibility wrapper around dataset_augment."""

from __future__ import annotations

from smartrain.services.datasets import dataset_augment as _impl

_detect_split = _impl._detect_split
_read_yolo_classes = _impl._read_yolo_classes
_parse_yolo_labels = _impl._parse_yolo_labels
_serialize_yolo_labels = _impl._serialize_yolo_labels
_sanitize_yolo_box = _impl._sanitize_yolo_box
_base36 = _impl._base36
_variant_code = _impl._variant_code
_aug_stem = _impl._aug_stem
_labels_signature_iou = _impl._labels_signature_iou
_collect_class_freq = _impl._collect_class_freq
_label_class_counts = _impl._label_class_counts
_labels_class_counts = _impl._labels_class_counts
_inserted_class_delta = _impl._inserted_class_delta
_image_soft_weight = _impl._image_soft_weight
_scaled_copies = _impl._scaled_copies
count_yolo_bbox_lines = _impl.count_yolo_bbox_lines
_train_split_class_bbox_counts = _impl._train_split_class_bbox_counts
_train_split_bbox_sum = _impl._train_split_bbox_sum
_image_tail_priority_score = _impl._image_tail_priority_score
_reorder_items_for_bbox_budget = _impl._reorder_items_for_bbox_budget
_class_aware_enabled = _impl._class_aware_enabled
_class_aware_trigger_prob = _impl._class_aware_trigger_prob
_effective_orthogonal_prob_geo = _impl._effective_orthogonal_prob_geo
_effective_flip_prob_geo = _impl._effective_flip_prob_geo
_geo_photo_trigger = _impl._geo_photo_trigger
_aug_extra_budget_allow = _impl._aug_extra_budget_allow
sum_train_bbox_disk = _impl.sum_train_bbox_disk
sum_train_class_bbox_disk = _impl.sum_train_class_bbox_disk
_provided_augment_flags = _impl._provided_augment_flags
_apply_augment_preset_defaults = _impl._apply_augment_preset_defaults
_warn_exhaustive_class_aware = _impl._warn_exhaustive_class_aware
_to_xyxy = _impl._to_xyxy
_to_yolo = _impl._to_yolo
_iou = _impl._iou
_center = _impl._center
_pick_uniform_grid_position = _impl._pick_uniform_grid_position
_roi_from_labels = _impl._roi_from_labels
_classify_side = _impl._classify_side
_inside = _impl._inside
_parse_roi_class_ids = _impl._parse_roi_class_ids
_detect_roi_box = _impl._detect_roi_box
_build_detector_roi_cache = _impl._build_detector_roi_cache
_apply_geom_aug = _impl._apply_geom_aug
_apply_exact_center_rotate = _impl._apply_exact_center_rotate

__all__ = [
    "_detect_split",
    "_read_yolo_classes",
    "_parse_yolo_labels",
    "_serialize_yolo_labels",
    "_sanitize_yolo_box",
    "_base36",
    "_variant_code",
    "_aug_stem",
    "_labels_signature_iou",
    "_collect_class_freq",
    "_label_class_counts",
    "_labels_class_counts",
    "_inserted_class_delta",
    "_image_soft_weight",
    "_scaled_copies",
    "count_yolo_bbox_lines",
    "_train_split_class_bbox_counts",
    "_train_split_bbox_sum",
    "_image_tail_priority_score",
    "_reorder_items_for_bbox_budget",
    "_class_aware_enabled",
    "_class_aware_trigger_prob",
    "_effective_orthogonal_prob_geo",
    "_effective_flip_prob_geo",
    "_geo_photo_trigger",
    "_aug_extra_budget_allow",
    "sum_train_bbox_disk",
    "sum_train_class_bbox_disk",
    "_provided_augment_flags",
    "_apply_augment_preset_defaults",
    "_warn_exhaustive_class_aware",
    "_to_xyxy",
    "_to_yolo",
    "_iou",
    "_center",
    "_pick_uniform_grid_position",
    "_roi_from_labels",
    "_classify_side",
    "_inside",
    "_parse_roi_class_ids",
    "_detect_roi_box",
    "_build_detector_roi_cache",
    "_apply_geom_aug",
    "_apply_exact_center_rotate",
]
