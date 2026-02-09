import math
from dataclasses import dataclass

import torch


@dataclass
class TAARConfig:
    eta_feature_attn: float = 0.8
    eta_sample_attn: float = 0.8
    eta_cont: float = 0.2
    feature_max_ratio: float = 0.3
    min_features: int = 1
    max_features: int | None = None
    max_samples: int | None = 2000
    features_per_group: int = 2

    @classmethod
    def from_retrieval_config(cls, retrieval_config: dict | None) -> "TAARConfig":
        retrieval_config = retrieval_config or {}
        return cls(
            eta_feature_attn=float(retrieval_config.get("eta_feature_attn", 0.8)),
            eta_sample_attn=float(retrieval_config.get("eta_sample_attn", retrieval_config.get("threshold", 0.8))),
            eta_cont=float(retrieval_config.get("eta_cont", retrieval_config.get("dynamic_ratio", 0.2))),
            feature_max_ratio=float(retrieval_config.get("feature_max_ratio", 0.3)),
            min_features=int(retrieval_config.get("min_features", 1)),
            max_features=retrieval_config.get("max_features"),
            max_samples=retrieval_config.get("max_samples", 2000),
            features_per_group=int(retrieval_config.get("features_per_group", 2)),
        )


class TaskAlignedAttentionRetrieval:
    @staticmethod
    def _ensure_2d(sample_attention: torch.Tensor) -> torch.Tensor:
        if sample_attention.dim() == 3:
            if sample_attention.shape[0] == 1:
                return sample_attention[0]
            return sample_attention.mean(dim=0)
        if sample_attention.dim() != 2:
            raise ValueError(f"Expected 2D/3D sample attention, but got shape={tuple(sample_attention.shape)}")
        return sample_attention

    @staticmethod
    def _num_elements_for_mass(scores: torch.Tensor, mass_ratio: float) -> int:
        if scores.numel() == 0:
            return 0
        mass_ratio = float(max(0.0, min(1.0, mass_ratio)))
        sorted_values, _ = torch.sort(scores, descending=True)
        total_mass = torch.sum(sorted_values)
        if total_mass <= 0:
            return 1
        target_mass = total_mass * mass_ratio
        cumsum = torch.cumsum(sorted_values, dim=0)
        idx = int(torch.searchsorted(cumsum, target_mass, right=False).item()) + 1
        return max(1, min(idx, scores.numel()))

    @staticmethod
    def select_sample_indices(
        sample_attention: torch.Tensor,
        config: TAARConfig,
    ) -> list[torch.Tensor]:
        score_2d = TaskAlignedAttentionRetrieval._ensure_2d(sample_attention)
        n_test, n_train = score_2d.shape
        k_context = max(1, int(math.ceil(config.eta_cont * n_train)))
        sorted_indices = torch.argsort(score_2d, dim=1, descending=True)
        top_indices: list[torch.Tensor] = []
        for row_id in range(n_test):
            row_scores = score_2d[row_id]
            k_attn = TaskAlignedAttentionRetrieval._num_elements_for_mass(row_scores, config.eta_sample_attn)
            k = max(k_attn, k_context)
            if config.max_samples is not None:
                k = min(k, int(config.max_samples))
            k = max(1, min(k, n_train))
            top_indices.append(sorted_indices[row_id, :k].to(sample_attention.device))
        return top_indices

    @staticmethod
    def _extract_group_feature_scores(feature_attention: torch.Tensor) -> torch.Tensor:
        if feature_attention.dim() == 2:
            feature_attention = feature_attention.unsqueeze(0)
        if feature_attention.dim() != 3:
            raise ValueError(
                f"Expected 2D/3D feature attention, but got shape={tuple(feature_attention.shape)}"
            )
        if feature_attention.shape[1] < 2 or feature_attention.shape[2] < 2:
            raise ValueError("Feature attention tensor is too small to contain a target token and feature tokens.")
        # Attention from target token to feature tokens.
        return feature_attention[:, -1, :-1]

    @staticmethod
    def _expand_group_scores(
        group_scores: torch.Tensor,
        num_raw_features: int,
        features_per_group: int,
    ) -> torch.Tensor:
        if features_per_group <= 1:
            expanded = group_scores
        else:
            expanded = torch.repeat_interleave(group_scores, repeats=features_per_group, dim=1)
        if expanded.shape[1] < num_raw_features:
            pad_size = num_raw_features - expanded.shape[1]
            expanded = torch.nn.functional.pad(expanded, (0, pad_size), mode="constant", value=0.0)
        return expanded[:, :num_raw_features]

    @staticmethod
    def select_feature_indices(
        feature_attention: torch.Tensor,
        num_raw_features: int,
        config: TAARConfig,
    ) -> torch.Tensor:
        group_scores = TaskAlignedAttentionRetrieval._extract_group_feature_scores(feature_attention)
        expanded_scores = TaskAlignedAttentionRetrieval._expand_group_scores(
            group_scores=group_scores,
            num_raw_features=num_raw_features,
            features_per_group=config.features_per_group,
        )
        global_scores = expanded_scores.mean(dim=0)
        sorted_scores, sorted_indices = torch.sort(global_scores, descending=True)
        k_attn = TaskAlignedAttentionRetrieval._num_elements_for_mass(sorted_scores, config.eta_feature_attn)
        k_prop = max(1, int(math.ceil(config.feature_max_ratio * num_raw_features)))
        k = min(k_attn, k_prop)
        k = max(k, int(config.min_features))
        if config.max_features is not None:
            k = min(k, int(config.max_features))
        k = max(1, min(k, num_raw_features))
        return sorted_indices[:k]
