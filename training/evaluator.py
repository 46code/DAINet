"""Validation / test evaluator for DAINet.

Calls ``model(rgb)`` only — null-conditioned, RGB-only — so the metrics it
produces match the test/benchmark scenario.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from evaluation.failure_analysis import FailureBuckets
from evaluation.metrics import MetricComputer


class Evaluator:
    def __init__(self, model: torch.nn.Module, loss_fn=None, device: str = "cuda"):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        device: str | None = None,
        with_lpips: bool = True,
        max_batches: int | None = None,
        failure_buckets: FailureBuckets | None = None,
    ) -> dict[str, float]:
        device = device or self.device
        was_training = self.model.training
        self.model.eval()
        metrics = MetricComputer(with_lpips=with_lpips)
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = self.model(batch["input_rgb"])
            metrics.update(
                out["output"],
                batch["target"],
                mask=batch.get("probe_mask"),
            )
            if failure_buckets is not None and "phi" in batch and "theta" in batch:
                failure_buckets.update(
                    out["output"],
                    batch["target"],
                    phi=batch["phi"],
                    theta=batch["theta"],
                    scenes=batch.get("scene"),
                )
        if was_training:
            self.model.train()
        return metrics.compute()
