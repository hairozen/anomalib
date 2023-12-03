"""Callback that loads model weights from the state dict."""

from __future__ import annotations

import logging

import torch
from pytorch_lightning import Trainer

from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks import LoadModelCallback

logger = logging.getLogger(__name__)


class AiVadLoadModelCallback(LoadModelCallback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path) -> None:
        super().__init__(weights_path)

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Call when inference begins.

        1. Loads the model weights from ``weights_path`` into the PyTorch module.
        1. Loads the estimators memory banks from ``weights_path`` into the Density module.
        """
        super().setup(trainer, pl_module, stage)

        weights = torch.load(self.weights_path, map_location=pl_module.device)

        logger.info("Loading the estimators memory banks from %s", self.weights_path)

        if "velocity_estimator_memory_bank" in weights:
            pl_module.model.density_estimator.velocity_estimator.memory_bank = weights["velocity_estimator_memory_bank"]
        if "pose_estimator_memory_bank" in weights:
            pl_module.model.density_estimator.pose_estimator.memory_bank = weights["pose_estimator_memory_bank"]
        if "appearance_estimator_memory_bank" in weights:
            pl_module.model.density_estimator.appearance_estimator.memory_bank = weights["appearance_estimator_memory_bank"]

